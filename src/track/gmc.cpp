#include "gmc.h"

#include <iostream>
#include <opencv2/core/eigen.hpp>

#include "inireader.h"

namespace botsort {
std::unique_ptr<GMCAlgorithm> GMCAlgorithm::createAlgo(const std::string& gmc_name,
                                                       const std::string& config_path) {
    if (gmc_name == "orb") {
        return std::make_unique<OrbGMC>(config_path);
    } else {
        std::cerr << "Unsupport gmc method" << std::endl;
        exit(1);
    }
}

OrbGMC::OrbGMC(const std::string& config_path) {
    loadParamsFromINI(config_path);

    detector_ = cv::FastFeatureDetector::create();
    extractor_ = cv::ORB::create();
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
}

HomoGraphyMatrix OrbGMC::apply(const cv::Mat& frame_raw, const std::vector<Detection>& detections) {
    int height = frame_raw.rows;
    int width = frame_raw.cols;
    cv::Mat gray_frame;
    cv::cvtColor(frame_raw, gray_frame, cv::COLOR_BGR2GRAY);

    if (downscale_ > 1.0) {
        width /= downscale_;
        height /= downscale_;
        cv::resize(gray_frame, gray_frame, cv::Size(width, height));
    }
    cv::Mat mask = createMask(gray_frame, detections);

    std::vector<cv::KeyPoint> keypoints;
    detector_->detect(gray_frame, keypoints, mask);
    cv::Mat descriptors;
    extractor_->compute(gray_frame, keypoints, descriptors);

    HomoGraphyMatrix H = HomoGraphyMatrix::Identity();
    if (!first_frame_initialized_) {
        first_frame_initialized_ = true;
        prev_frame_ = gray_frame.clone();
        prev_keypoints_ = keypoints;
        prev_descriptors_ = descriptors.clone();
        return H;
    }

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(prev_descriptors_, descriptors, knn_matches, 2);
    std::vector<cv::DMatch> matches;
    std::vector<cv::Point2f> spatial_distances;
    cv::Point2f max_spatial_distance(0.25f * width, 0.25f * height);

    for (const auto& knn_match : knn_matches) {
        const auto& m = knn_match[0];
        const auto& n = knn_match[1];

        if (m.distance < 0.9 * n.distance) {
            cv::Point2f prev_keypoint_location = prev_keypoints_[m.queryIdx].pt;
            cv::Point2f curr_keypoint_location = keypoints[m.trainIdx].pt;
            cv::Point2f distance = prev_keypoint_location - curr_keypoint_location;
            if (cv::abs(distance.x) < max_spatial_distance.x &&
                cv::abs(distance.y) < max_spatial_distance.y) {
                spatial_distances.push_back(distance);
                matches.push_back(m);
            }
        }
    }
    if (matches.empty()) {
        prev_frame_ = gray_frame.clone();
        prev_keypoints_ = keypoints;
        prev_descriptors_ = descriptors.clone();
        return H;
    }

    cv::Scalar mean_spatial_distance, std_spatial_distance;
    cv::meanStdDev(spatial_distances, mean_spatial_distance, std_spatial_distance);

    // Get good matches
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> prev_points, curr_points;
    for (size_t i = 0; i < matches.size(); ++i) {
        cv::Point2f mean_normalized_sd(spatial_distances[i].x - mean_spatial_distance[0],
                                       spatial_distances[i].y - mean_spatial_distance[1]);
        if (mean_normalized_sd.x < 2.5 * std_spatial_distance[0] &&
            mean_normalized_sd.y < 2.6 * std_spatial_distance[1]) {
            prev_points.push_back(prev_keypoints_[matches[i].queryIdx].pt);
            curr_points.push_back(keypoints[matches[i].trainIdx].pt);
        }
    }
    // Find transMat between previous fram and current frame
    if (prev_points.size() > 4) {
        cv::Mat inliers;
        cv::Mat homography = cv::findHomography(prev_points, curr_points, cv::RANSAC, 3, inliers,
                                                ransac_max_iters_, ransac_conf_);
        double inlier_ratio = cv::countNonZero(inliers) / static_cast<double>(inliers.rows);
        if (inlier_ratio > inlier_ratio_) {
            cv::cv2eigen(homography, H);
            if (downscale_ > 1.0) {
                H(0, 2) *= downscale_;
                H(1, 2) *= downscale_;
            }
        } else {
            std::cout << "Wraning: Could not estimate affine matrix" << std::endl;
        }
    }
    prev_frame_ = gray_frame.clone();
    prev_keypoints_ = keypoints;
    prev_descriptors_ = descriptors.clone();
    return H;
}

cv::Mat OrbGMC::createMask(const cv::Mat& frame, const std::vector<Detection>& detections) {
    cv::Mat mask = cv::Mat::zeros(frame.size(), frame.type());
    int height = frame.rows, width = frame.cols;
    // 边缘为不感兴趣区域
    cv::Rect roi(static_cast<int>(width * 0.02), static_cast<int>(height * 0.02),
                 static_cast<int>(width * 0.96), static_cast<int>(height * 0.96));
    mask(roi) = 255;
    // 去除目标框所在区域
    for (const auto& det : detections) {
        cv::Rect box(static_cast<int>(det.bbox_tlwh.x / downscale_),
                     static_cast<int>(det.bbox_tlwh.y / downscale_),
                     static_cast<int>(det.bbox_tlwh.width / downscale_),
                     static_cast<int>(det.bbox_tlwh.height / downscale_));
        mask(box) = 0;
    }
    return mask;
}

void OrbGMC::loadParamsFromINI(const std::string& config_path) {
    INIReader gmc_config(config_path);
    if (gmc_config.ParseError() < 0) {
        std::cerr << "Can't load " << config_path << std::endl;
        exit(1);
    }

    downscale_ = gmc_config.GetFloat(algo_name_, "downscale", 2.0);
    inlier_ratio_ = gmc_config.GetFloat(algo_name_, "inlier_ratio", 0.5);
    ransac_conf_ = gmc_config.GetFloat(algo_name_, "ransac_conf", 0.99);
    ransac_max_iters_ = gmc_config.GetInteger(algo_name_, "ransac_max_iters", 500);
}
} // namespace botsort