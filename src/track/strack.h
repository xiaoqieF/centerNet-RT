#ifndef CENTERRT_TRACK_STRACK_H
#define CENTERRT_TRACK_STRACK_H

#include <cstdint>
#include <memory>
#include <vector>

#include "kalmanfilter.h"

namespace botsort {
class STrack {
public:
    enum TrackState { New = 0, Tracked, Lost, LongLost, Removed };
    STrack(const std::vector<float> tlwh, float score, uint8_t class_id)
        : tlwh_(tlwh),
          state_(New),
          is_activated_(false),
          tracklet_len_(0),
          score_(score),
          class_id_(class_id) {}
    static int nextId();
    void activate(KalmanFilter& kalman_filter, uint32_t frame_id);
    void reActivate(KalmanFilter& kalman_filter, STrack& new_track, uint32_t frame_id);
    void predict(KalmanFilter& kalman_filter);
    static void multiPredict(std::vector<std::shared_ptr<STrack>>& tracks,
                             KalmanFilter& kalman_filter);
    void update(KalmanFilter& kalman_filter, STrack& new_track, uint32_t frame_id);
    void applyCameraMotion(const HomoGraphyMatrix& H);
    static void multiGMC(std::vector<std::shared_ptr<STrack>>& tracks, const HomoGraphyMatrix& H);

    bool isActivated() const { return is_activated_; }
    int trackId() const { return track_id_; }
    const std::vector<float>& tlwh() const { return tlwh_; }
    float score() const { return score_; }
    TrackState state() const { return state_; }
    void markLost() { state_ = Lost; }
    void markRemoved() { state_ = Removed; }
    uint32_t endFrame() const { return frame_id_; }

private:
    void updateTrackletTlwh();
    std::vector<float> tlwh_;
    TrackState state_;
    bool is_activated_;
    uint32_t tracklet_len_;
    float score_;
    uint8_t class_id_;
    int track_id_;
    uint32_t frame_id_;
    uint32_t start_frame_;
    KalmanFilter::StateSpaceVec mean_;
    KalmanFilter::ErrorCovMatrix covariance_;
};

} // namespace botsort

#endif