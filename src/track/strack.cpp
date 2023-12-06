#include "strack.h"

#include "datatype.h"

namespace botsort {
namespace detail {
void buildDetVecFromTlwh(DetVec& det, const std::vector<float>& tlwh) {
    det << tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2, tlwh[2], tlwh[3];
}
} // namespace detail

int STrack::nextId() {
    static int _count = 0;
    _count++;
    return _count;
}

void STrack::activate(KalmanFilter& kalman_filter, uint32_t frame_id) {
    track_id_ = nextId();
    DetVec detection_box;
    detail::buildDetVecFromTlwh(detection_box, tlwh_);
    auto state_space = kalman_filter.init(detection_box);
    mean_ = state_space.first;
    covariance_ = state_space.second;

    if (frame_id == 1) {
        is_activated_ = true;
    }
    frame_id_ = frame_id;
    start_frame_ = frame_id;
    state_ = Tracked;
    tracklet_len_ = 1;
    updateTrackletTlwh();
}

void STrack::reActivate(KalmanFilter& kalman_filter, STrack& new_track, uint32_t frame_id) {
    DetVec new_track_box;
    detail::buildDetVecFromTlwh(new_track_box, new_track.tlwh_);

    auto state_space = kalman_filter.update(mean_, covariance_, new_track_box);
    mean_ = state_space.first;
    covariance_ = state_space.second;

    tracklet_len_ = 0;
    state_ = Tracked;
    is_activated_ = true;
    score_ = new_track.score_;
    frame_id_ = frame_id;
    updateTrackletTlwh();
}

void STrack::predict(KalmanFilter& kalman_filter) {
    // set velocity of w and h to 0
    if (state_ != Tracked) {
        mean_(6) = mean_(7) = 0;
    }
    kalman_filter.predict(mean_, covariance_);
    updateTrackletTlwh();
}

void STrack::multiPredict(std::vector<std::shared_ptr<STrack>>& tracks,
                          KalmanFilter& kalman_filter) {
    for (auto& track : tracks) {
        track->predict(kalman_filter);
    }
}

void STrack::update(KalmanFilter& kalman_filter, STrack& new_track, uint32_t frame_id) {
    DetVec new_track_box;
    detail::buildDetVecFromTlwh(new_track_box, new_track.tlwh_);

    auto state_space = kalman_filter.update(mean_, covariance_, new_track_box);
    mean_ = state_space.first;
    covariance_ = state_space.second;
    state_ = Tracked;
    is_activated_ = true;
    score_ = new_track.score_;
    tracklet_len_++;
    frame_id_ = frame_id;
}

void STrack::applyCameraMotion(const HomoGraphyMatrix& H) {
    Eigen::Matrix<float, 8, 8> R8x8 = Eigen::Matrix<float, 8, 8>::Identity();
    // 和原实现不一样
    for (int i = 0; i < 4; ++i) {
        R8x8.block(2 * i, 2 * i, 2, 2) = H.block<2, 2>(0, 0);
    }
    // X = MX^- + T
    mean_ = R8x8 * mean_;
    mean_.head(2) += H.block<2, 1>(0, 2);

    covariance_ = R8x8 * covariance_ * R8x8.transpose();
}

void STrack::multiGMC(std::vector<std::shared_ptr<STrack>>& tracks, const HomoGraphyMatrix& H) {
    for (auto& track : tracks) {
        track->applyCameraMotion(H);
    }
}

void STrack::updateTrackletTlwh() {
    tlwh_ = {mean_(0) - mean_(2) / 2, mean_(1) - mean_(3) / 2, mean_(2), mean_(3)};
}

} // namespace botsort