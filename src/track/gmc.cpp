#include "gmc.h"

#include <iostream>
#include <opencv2/core/eigen.hpp>

#include "inireader.h"

namespace botsort {
std::unique_ptr<GMCAlgorithm> GMCAlgorithm::createAlgo(const std::string& gmc_name,
                                                       const std::string& config_path) {
    if (gmc_name == "orb") {
        return std::make_unique<OrbGMC>(config_path);
    } else if (gmc_name == "sparseOptFlow") {
        return std::make_unique<SparseOptFlowGMC>(config_path);
    } else {
        std::cerr << "Do not use gmc method" << std::endl;
        return nullptr;
    }
}

OrbGMC::OrbGMC(const std::string& config_path) {
    loadParamsFromINI(config_path);

    detector_ = cv::FastFeatureDetector::create();
    extractor_ = cv::ORB::create();
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
}

HomoGraphyMatrix OrbGMC::apply(const cv::Mat& frame_raw,
                               const std::vector<common::Detection>& detections) {
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
            mean_normalized_sd.y < 2.5 * std_spatial_distance[1]) {
            prev_points.push_back(prev_keypoints_[matches[i].queryIdx].pt);
            curr_points.push_back(keypoints[matches[i].trainIdx].pt);
            good_matches.push_back(matches[i]);
        }
    }
    // Find transMat between previous fram and current frame
    if (prev_points.size() > 4) {
        cv::Mat inliers;
        cv::Mat homography = cv::findHomography(prev_points, curr_points, cv::RANSAC, 3, inliers,
                                                ransac_max_iters_, ransac_conf_);
        std::cout << "Homography is: " << homography << std::endl;
        double inlier_ratio = cv::countNonZero(inliers) / static_cast<double>(inliers.rows);
        if (inlier_ratio > inlier_ratio_) {
            cv::cv2eigen(homography, H);
            std::cout << "H is: " << H << std::endl;
            if (downscale_ > 1.0) {
                H(0, 2) *= downscale_;
                H(1, 2) *= downscale_;
            }
        } else {
            std::cout << "Wraning: Could not estimate affine matrix" << std::endl;
        }
    }
    // for debug
    // cv::Mat matches_img;
    // cv::hconcat(prev_frame_, gray_frame, matches_img);
    // cv::cvtColor(matches_img, matches_img, cv::COLOR_GRAY2BGR);

    // int W = prev_frame_.cols;

    // for (const auto& m : good_matches) {
    //     cv::Point prev_pt = prev_keypoints_[m.queryIdx].pt;
    //     cv::Point curr_pt = keypoints[m.trainIdx].pt;

    //     curr_pt.x += W;

    //     cv::Scalar color = cv::Scalar::all(rand() % 255);
    //     color = cv::Scalar((int)color[0], (int)color[1], (int)color[2]);

    //     cv::line(matches_img, prev_pt, curr_pt, color, 1, cv::LineTypes::LINE_AA);
    //     cv::circle(matches_img, prev_pt, 2, color, -1);
    //     cv::circle(matches_img, curr_pt, 2, color, -1);
    // }

    // cv::imshow("Matches", matches_img);
    // cv::waitKey(0);

    prev_frame_ = gray_frame.clone();
    prev_keypoints_ = keypoints;
    prev_descriptors_ = descriptors.clone();
    return H;
}

cv::Mat OrbGMC::createMask(const cv::Mat& frame, const std::vector<common::Detection>& detections) {
    cv::Mat mask = cv::Mat::zeros(frame.size(), frame.type());
    int height = frame.rows, width = frame.cols;
    // 边缘为不感兴趣区域
    cv::Rect roi(static_cast<int>(width * 0.02), static_cast<int>(height * 0.02),
                 static_cast<int>(width * 0.96), static_cast<int>(height * 0.96));
    mask(roi) = 255;
    // 去除目标框所在区域
    for (const auto& det : detections) {
        cv::Rect box(static_cast<int>(det.box.x1 / downscale_),
                     static_cast<int>(det.box.y1 / downscale_),
                     static_cast<int>((det.box.x2 - det.box.x1) / downscale_),
                     static_cast<int>((det.box.y2 - det.box.y1) / downscale_));
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

SparseOptFlowGMC::SparseOptFlowGMC(const std::string& config_path) {
    loadParamsFromINI(config_path);
}

HomoGraphyMatrix SparseOptFlowGMC::apply(const cv::Mat& frame_raw,
                                         const std::vector<common::Detection>& detections) {
    // Initialization
    int height = frame_raw.rows;
    int width = frame_raw.cols;

    HomoGraphyMatrix H;
    H.setIdentity();

    cv::Mat frame;
    cv::cvtColor(frame_raw, frame, cv::COLOR_BGR2GRAY);

    // Downscale
    if (downscale_ > 1.0f) {
        width /= downscale_, height /= downscale_;
        cv::resize(frame, frame, cv::Size(width, height));
    }

    // Detect keypoints
    std::vector<cv::Point2f> keypoints;
    cv::goodFeaturesToTrack(frame, keypoints, max_corners_, quality_level_, min_distance_,
                            cv::noArray(), block_size_, use_harris_detector_, k_);

    if (!first_frame_initialized_ || prev_keypoints_.size() == 0) {
        /**
         *  If this is the first frame, there is nothing to match
         *  Save the keypoints and descriptors, return identity matrix
         */
        first_frame_initialized_ = true;
        prev_frame_ = frame.clone();
        prev_keypoints_ = keypoints;
        return H;
    }

    // Find correspondences between the previous and current frame
    std::vector<cv::Point2f> matched_keypoints;
    std::vector<uchar> status;
    std::vector<float> err;
    try {
        cv::calcOpticalFlowPyrLK(prev_frame_, frame, prev_keypoints_, matched_keypoints, status,
                                 err);
    } catch (const cv::Exception& e) {
        std::cout << "Warning: Could not find correspondences for GMC" << std::endl;
        return H;
    }

    // Keep good matches
    std::vector<cv::Point2f> prev_points, curr_points;
    for (size_t i = 0; i < matched_keypoints.size(); i++) {
        if (status[i]) {
            prev_points.push_back(prev_keypoints_[i]);
            curr_points.push_back(matched_keypoints[i]);
        }
    }

    // Estimate affine matrix
    if (prev_points.size() > 4) {
        cv::Mat inliers;
        cv::Mat homography = cv::findHomography(prev_points, curr_points, cv::RANSAC, 3, inliers,
                                                ransac_max_iters_, ransac_conf_);

        double inlier_ratio = cv::countNonZero(inliers) / (double)inliers.rows;
        if (inlier_ratio > inlier_ratio_) {
            cv2eigen(homography, H);
            if (downscale_ > 1.0) {
                H(0, 2) *= downscale_;
                H(1, 2) *= downscale_;
            }
        } else {
            std::cout << "Warning: Could not estimate affine matrix" << std::endl;
        }
    }

    prev_frame_ = frame.clone();
    prev_keypoints_ = keypoints;
    return H;
}

void SparseOptFlowGMC::loadParamsFromINI(const std::string& config_path) {
    INIReader gmc_config(config_path);
    if (gmc_config.ParseError() < 0) {
        std::cout << "Can't load " << config_path << std::endl;
        exit(1);
    }

    use_harris_detector_ = gmc_config.GetBoolean(algo_name_, "use_harris_detector", false);

    max_corners_ = gmc_config.GetInteger(algo_name_, "max_corners", 1000);
    block_size_ = gmc_config.GetInteger(algo_name_, "block_size", 3);
    ransac_max_iters_ = gmc_config.GetInteger(algo_name_, "ransac_max_iters", 500);

    quality_level_ = gmc_config.GetReal(algo_name_, "quality_level", 0.01);
    k_ = gmc_config.GetReal(algo_name_, "k", 0.04);
    min_distance_ = gmc_config.GetReal(algo_name_, "min_distance", 1.0);

    downscale_ = gmc_config.GetFloat(algo_name_, "downscale", 2.0F);
    inlier_ratio_ = gmc_config.GetFloat(algo_name_, "inlier_ratio", 0.5);
    ransac_conf_ = gmc_config.GetFloat(algo_name_, "ransac_conf", 0.99);
}
} // namespace botsort