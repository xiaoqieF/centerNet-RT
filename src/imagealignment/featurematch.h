#ifndef CENTERRT_SIFT_H
#define CENTERRT_SIFT_H

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "common/datatype.h"

namespace imgalign {
class FeatureMatch {
public:
    enum Method { ORB = 0, SIFT = 1 };
    FeatureMatch(const std::string& img_path1, const std::string& img_path2, Method method);
    void match(const std::vector<common::Detection>& mask1,
               const std::vector<common::Detection>& mask2);

private:
    cv::Mat createMask(const cv::Mat& frame, const std::vector<common::Detection>& detections);
    cv::Mat src1_;
    cv::Mat src2_;
    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
};
} // namespace imgalign

#endif