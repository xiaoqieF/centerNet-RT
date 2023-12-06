#ifndef CENTERRT_BOTSORT_H
#define CENTERRT_BOTSORT_H

#include <memory>
#include <string>
#include <vector>

#include "gmc.h"
#include "kalmanfilter.h"
#include "strack.h"

namespace botsort {
class BoTSORT {
public:
    using STrackList = std::vector<std::shared_ptr<STrack>>;
    explicit BoTSORT(const std::string& config_path);
    STrackList track(const std::vector<Detection>& dets, const cv::Mat& frame);

private:
    void loadParamsFromINI(const std::string& config_path);
    std::pair<STrackList, STrackList> splitDetectionsByConfidence(
        const std::vector<Detection>& detections);
    static STrackList mergeTrackLists(const STrackList& list_a, const STrackList& list_b);
    static STrackList removeFromList(const STrackList& tracks, const STrackList& to_remove);

    std::string gmc_name_;
    uint8_t track_buffer_;
    uint8_t frame_rate_;
    uint8_t buffer_size_;
    uint8_t max_time_lost_;
    float track_high_thresh_;
    float track_low_thresh_;
    float new_track_thresh_;
    float match_thresh_;
    float proximity_thresh_;
    float apperance_thresh_;
    float lambda_;
    uint32_t frame_id_;

    STrackList tracked_tracks_;
    STrackList lost_tracks_;
    std::unique_ptr<KalmanFilter> kalman_filter_;
    std::unique_ptr<GMCAlgorithm> gmc_algo_;
};

} // namespace botsort

#endif