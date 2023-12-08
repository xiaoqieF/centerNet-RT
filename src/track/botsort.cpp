#include "botsort.h"

#include <iostream>

#include "inireader.h"
#include "matching.h"

namespace botsort {
BoTSORT::BoTSORT(const std::string& config_path) {
    loadParamsFromINI(config_path);
    frame_id_ = 0;
    buffer_size_ = static_cast<uint8_t>(frame_rate_ / 30.0 * track_buffer_);
    max_time_lost_ = buffer_size_;
    // kalman_filter_ = std::make_unique<KalmanFilter>(static_cast<double>(1.0 / frame_rate_));
    kalman_filter_ = std::make_unique<KalmanFilter>(1.0);
    gmc_algo_ = GMCAlgorithm::createAlgo(gmc_name_, "../src/track/config/gmc.ini");
}

BoTSORT::STrackList BoTSORT::track(const std::vector<common::Detection>& detections,
                                   const cv::Mat& frame) {
    frame_id_++;
    // 为所有检测框生成 STrack 对象，并将其分为高分框和低分框
    auto [detections_high_conf, detections_low_conf] = splitDetectionsByConfidence(detections);
    // 将已有轨迹分为 unconfirmed 和 tracked
    STrackList unconfirmed_tracks, tracked_tracks;
    for (const auto& track : tracked_tracks_) {
        if (!track->isActivated()) {
            unconfirmed_tracks.push_back(track);
        } else {
            tracked_tracks.push_back(track);
        }
    }
    STrackList tracks_pool = mergeTrackLists(tracked_tracks, lost_tracks_);
    // 已确认的所有轨迹(包括暂时丢失的)都进行 Kalman 预测
    STrack::multiPredict(tracks_pool, *kalman_filter_);
    // HomoGraphyMatrix H = gmc_algo_->apply(frame, detections);
    // STrack::multiGMC(tracks_pool, H);
    // STrack::multiGMC(unconfirmed_tracks, H);

    // 1. 对高分框和轨迹进行 iou 匹配
    // 将高分检测框 detection_high_conf 和 所有已确认的轨迹(确认的和暂时丢失的) tracks_pool 进行匹配
    CostMatrix iou_dists = matching::iouDistance(tracks_pool, detections_high_conf);
    matching::fuseScore(iou_dists, detections_high_conf);
    AssociationData first_associations = matching::lineraAssignment(iou_dists, match_thresh_);

    STrackList activated_tracks, refind_tracks;
    for (auto match : first_associations.matches) {
        std::shared_ptr<STrack>& track = tracks_pool[match.first];
        std::shared_ptr<STrack>& detection = detections_high_conf[match.second];

        if (track->state() == STrack::Tracked) {
            track->update(*kalman_filter_, *detection, frame_id_);
            activated_tracks.push_back(track);
        } else {
            track->reActivate(*kalman_filter_, *detection, frame_id_);
            refind_tracks.push_back(track);
        }
    }
    // 2. 对低分检测框进行匹配
    // 将低分检测框 detections_low_conf 和第一次匹配剩余的轨迹进行匹配
    STrackList unmatched_tracks_after_1st_association;
    for (int track_idx : first_associations.unmatched_track_indices) {
        std::shared_ptr<STrack>& track = tracks_pool[track_idx];
        if (track->state() == STrack::Tracked) {
            unmatched_tracks_after_1st_association.push_back(track);
        }
    }
    CostMatrix iou_dists_second =
        matching::iouDistance(unmatched_tracks_after_1st_association, detections_low_conf);
    AssociationData second_associations = matching::lineraAssignment(iou_dists_second, 0.5);

    for (auto match : second_associations.matches) {
        std::shared_ptr<STrack>& track = unmatched_tracks_after_1st_association[match.first];
        std::shared_ptr<STrack>& detection = detections_low_conf[match.second];

        if (track->state() == STrack::Tracked) {
            track->update(*kalman_filter_, *detection, frame_id_);
            activated_tracks.push_back(track);
        } else {
            track->reActivate(*kalman_filter_, *detection, frame_id_);
            refind_tracks.push_back(track);
        }
    }
    // 两次匹配均未匹配上的轨迹标记为丢失
    STrackList lost_tracks;
    for (int unmatched_track_index : second_associations.unmatched_track_indices) {
        std::shared_ptr<STrack>& track =
            unmatched_tracks_after_1st_association[unmatched_track_index];
        if (track->state() != STrack::Lost) {
            track->markLost();
            lost_tracks.push_back(track);
        }
    }

    // 3. 处理 unconfirmed 轨迹，将其与剩余的高分检测框匹配
    STrackList unmatched_detections_after_1st_association;
    for (int detection_idx : first_associations.unmatched_det_indices) {
        unmatched_detections_after_1st_association.push_back(detections_high_conf[detection_idx]);
    }
    CostMatrix iou_dists_unconfirmed =
        matching::iouDistance(unconfirmed_tracks, unmatched_detections_after_1st_association);
    matching::fuseScore(iou_dists_unconfirmed, unmatched_detections_after_1st_association);
    AssociationData unconfirmed_associations =
        matching::lineraAssignment(iou_dists_unconfirmed, 0.7);

    for (auto match : unconfirmed_associations.matches) {
        std::shared_ptr<STrack>& track = unconfirmed_tracks[match.first];
        std::shared_ptr<STrack>& detection =
            unmatched_detections_after_1st_association[match.second];

        track->update(*kalman_filter_, *detection, frame_id_);
        activated_tracks.push_back(track);
    }
    STrackList removed_tracks;
    for (int unmatched_track_index : unconfirmed_associations.unmatched_track_indices) {
        std::shared_ptr<STrack>& track = unconfirmed_tracks[unmatched_track_index];
        track->markRemoved();
        removed_tracks.push_back(track);
    }

    // 4. 初始化新轨迹(多次匹配之后均未匹配上的高分检测框)
    for (int detection_idx : unconfirmed_associations.unmatched_det_indices) {
        std::shared_ptr<STrack>& detection =
            unmatched_detections_after_1st_association[detection_idx];
        if (detection->score() >= new_track_thresh_) {
            detection->activate(*kalman_filter_, frame_id_);
            activated_tracks.push_back(detection);
        }
    }

    // 5. 更新丢失轨迹的状态
    for (auto& track : lost_tracks_) {
        if (frame_id_ - track->endFrame() > max_time_lost_) {
            track->markRemoved();
            removed_tracks.push_back(track);
        }
    }

    STrackList updated_tracked_tracks;
    for (auto& track : tracked_tracks_) {
        if (track->state() == STrack::Tracked) {
            updated_tracked_tracks.push_back(track);
        }
    }
    /// TODO: 简化耗时操作
    tracked_tracks_ = mergeTrackLists(updated_tracked_tracks, activated_tracks);
    tracked_tracks_ = mergeTrackLists(tracked_tracks_, refind_tracks);

    lost_tracks_ = mergeTrackLists(lost_tracks_, lost_tracks);
    lost_tracks_ = removeFromList(lost_tracks_, tracked_tracks_);
    lost_tracks_ = removeFromList(lost_tracks_, removed_tracks);

    // 原实现还需要去除重复的 track

    STrackList output_tracks;
    for (auto& track : tracked_tracks_) {
        if (track->isActivated()) {
            output_tracks.push_back(track);
        }
    }
    return output_tracks;
}

void BoTSORT::loadParamsFromINI(const std::string& config_path) {
    INIReader tracker_config(config_path);
    if (tracker_config.ParseError() < 0) {
        std::cerr << "Can't load " << config_path << std::endl;
        exit(1);
    }
    const std::string tracker_name = "BoTSORT";
    track_high_thresh_ = tracker_config.GetFloat(tracker_name, "track_high_thresh", 0.6f);
    track_low_thresh_ = tracker_config.GetFloat(tracker_name, "track_low_thresh", 0.1f);
    new_track_thresh_ = tracker_config.GetFloat(tracker_name, "new_track_thresh", 0.7f);
    track_buffer_ = tracker_config.GetInteger(tracker_name, "track_buffer", 30);
    match_thresh_ = tracker_config.GetFloat(tracker_name, "match_thresh", 0.7f);
    proximity_thresh_ = tracker_config.GetFloat(tracker_name, "proximity_thresh", 0.5f);
    apperance_thresh_ = tracker_config.GetFloat(tracker_name, "apperance_thresh", 0.25f);
    gmc_name_ = tracker_config.Get(tracker_name, "gmc_method", "orb");
    frame_rate_ = tracker_config.GetInteger(tracker_name, "frame_rate", 30);
    lambda_ = tracker_config.GetFloat(tracker_name, "lambda", 0.985f);
}

std::pair<BoTSORT::STrackList, BoTSORT::STrackList> BoTSORT::splitDetectionsByConfidence(
    const std::vector<common::Detection>& detections) {
    std::vector<std::shared_ptr<STrack>> detections_high_conf, detections_low_conf;
    for (const common::Detection& detection : detections) {
        std::vector<float> tlwh{detection.box.x1, detection.box.y1,
                                detection.box.x2 - detection.box.x1,
                                detection.box.y2 - detection.box.y1};
        if (detection.prob > track_low_thresh_) {
            std::shared_ptr<STrack> tracklet =
                std::make_shared<STrack>(tlwh, detection.prob, detection.class_id);
            if (detection.prob >= track_high_thresh_) {
                detections_high_conf.push_back(tracklet);
            } else {
                detections_low_conf.push_back(tracklet);
            }
        }
    }
    return {detections_high_conf, detections_low_conf};
}

BoTSORT::STrackList BoTSORT::mergeTrackLists(const STrackList& list_a, const STrackList& list_b) {
    std::set<int> exists;
    STrackList res;
    for (const auto& track : list_a) {
        if (exists.count(track->trackId())) {
            continue;
        }
        exists.insert(track->trackId());
        res.push_back(track);
    }
    for (const auto& track : list_b) {
        if (exists.count(track->trackId())) {
            continue;
        }
        exists.insert(track->trackId());
        res.push_back(track);
    }
    return res;
}

BoTSORT::STrackList BoTSORT::removeFromList(const STrackList& tracks, const STrackList& to_remove) {
    std::set<int> exists;
    STrackList res;
    for (const auto& track : to_remove) {
        exists.insert(track->trackId());
    }
    for (const auto& track : tracks) {
        if (!exists.count(track->trackId())) {
            res.push_back(track);
        }
    }
    return res;
}

} // namespace botsort
