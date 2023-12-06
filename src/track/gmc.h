#ifndef CENTERRT_GMC_H
#define CENTERRT_GMC_H

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "datatype.h"

namespace botsort {
class GMCAlgorithm {
public:
    virtual ~GMCAlgorithm() = default;
    virtual HomoGraphyMatrix apply(const cv::Mat& frame_raw,
                                   const std::vector<Detection>& detections) = 0;
    static std::unique_ptr<GMCAlgorithm> createAlgo(const std::string& gmc_name,
                                                    const std::string& config_path);
};

class OrbGMC : public GMCAlgorithm {
public:
    explicit OrbGMC(const std::string& config_path);
    HomoGraphyMatrix apply(const cv::Mat& frame_raw,
                           const std::vector<Detection>& detections) override;

private:
    void loadParamsFromINI(const std::string& config_path);
    cv::Mat createMask(const cv::Mat& frame, const std::vector<Detection>& detections);
    std::string algo_name_ = "orb";
    float downscale_;
    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorExtractor> extractor_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;

    bool first_frame_initialized_ = false;
    cv::Mat prev_frame_;
    std::vector<cv::KeyPoint> prev_keypoints_;
    cv::Mat prev_descriptors_;
    float inlier_ratio_;
    float ransac_conf_;
    int ransac_max_iters_;
};

} // namespace botsort

#endif