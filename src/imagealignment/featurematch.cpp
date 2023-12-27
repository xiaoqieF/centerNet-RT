#include "featurematch.h"

#include <vector>

#include "common/datatype.h"

namespace imgalign {
FeatureMatch::FeatureMatch(const std::string& img_path1,
                           const std::string& img_path2,
                           Method method) {
    src1_ = cv::imread(img_path1);
    src2_ = cv::imread(img_path2);
    if (src1_.size() != src2_.size()) {
        std::cerr << "two image size should be same" << std::endl;
        exit(-1);
    }
    if (method == ORB) {
        detector_ = cv::ORB::create();
        matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
    } else if (method == SIFT) {
        detector_ = cv::SIFT::create();
        matcher_ = cv::BFMatcher::create(cv::NORM_L1);
    }
}

void FeatureMatch::match(const std::vector<common::Detection>& detections1,
                         const std::vector<common::Detection>& detections2) {
    cv::Mat gray_img1, gray_img2;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    cv::cvtColor(src1_, gray_img1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(src2_, gray_img2, cv::COLOR_BGR2GRAY);

    cv::Mat mask1 = createMask(gray_img1, detections1);
    cv::Mat mask2 = createMask(gray_img2, detections2);

    detector_->detectAndCompute(src1_, mask1, keypoints1, descriptors1);
    detector_->detectAndCompute(src2_, mask2, keypoints2, descriptors2);

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    std::vector<cv::DMatch> matches;
    std::vector<cv::Point2f> spatial_distances;
    cv::Point2f max_spatial_distance(0.25f * src1_.cols, 0.25f * src1_.rows);

    for (const auto& knn_match : knn_matches) {
        const auto& m = knn_match[0];
        const auto& n = knn_match[1];

        if (m.distance < 0.9 * n.distance) {
            cv::Point2f prev_keypoint_location = keypoints1[m.queryIdx].pt;
            cv::Point2f curr_keypoint_location = keypoints2[m.trainIdx].pt;
            cv::Point2f distance = prev_keypoint_location - curr_keypoint_location;
            if (cv::abs(distance.x) < max_spatial_distance.x &&
                cv::abs(distance.y) < max_spatial_distance.y) {
                spatial_distances.push_back(distance);
                matches.push_back(m);
            }
        }
    }
    if (matches.empty()) {
        std::cout << "no good match" << std::endl;
        return;
    }
    cv::Mat matches_img;
    cv::hconcat(src1_, src2_, matches_img);

    int W = gray_img1.cols;

    for (const auto& m : matches) {
        cv::Point prev_pt = keypoints1[m.queryIdx].pt;
        cv::Point curr_pt = keypoints2[m.trainIdx].pt;

        curr_pt.x += W;

        cv::Scalar color = cv::Scalar::all(rand() % 255);
        color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);

        cv::line(matches_img, prev_pt, curr_pt, color, 1, cv::LineTypes::LINE_AA);
        cv::circle(matches_img, prev_pt, 4, color, -1);
        cv::circle(matches_img, curr_pt, 4, color, -1);
    }

    cv::namedWindow("Matches", cv::WINDOW_FREERATIO);
    cv::imshow("Matches", matches_img);
    cv::waitKey(0);
}

cv::Mat FeatureMatch::createMask(const cv::Mat& frame,
                                 const std::vector<common::Detection>& detections) {
    cv::Mat mask = cv::Mat::zeros(frame.size(), frame.type());
    int height = frame.rows, width = frame.cols;
    // 边缘为不感兴趣区域
    cv::Rect roi(static_cast<int>(width * 0.02), static_cast<int>(height * 0.02),
                 static_cast<int>(width * 0.96), static_cast<int>(height * 0.96));
    mask(roi) = 255;
    // 去除目标框所在区域
    for (const auto& det : detections) {
        cv::Rect box(static_cast<int>(det.box.x1), static_cast<int>(det.box.y1),
                     static_cast<int>(det.box.x2 - det.box.x1),
                     static_cast<int>(det.box.y2 - det.box.y1));
        mask(box) = 0;
    }
    return mask;
}
} // namespace imgalign