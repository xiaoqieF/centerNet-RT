#include <iostream>

#include "det/centernet.h"
#include "det/centerutils.h"
#include "imagealignment/featurematch.h"

using namespace centernet;

void matchECC(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat img1_gray, img2_gray;
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);
    // 设置ECC配准的参数
    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 1000, 1e-5);
    cv::Mat warpMatrix = cv::Mat::eye(2, 3, CV_32F);

    // 执行ECC配准
    cv::findTransformECC(img1_gray, img2_gray, warpMatrix, cv::MOTION_EUCLIDEAN, criteria);

    // 应用配准变换
    cv::Mat alignedImg;
    cv::warpAffine(img2, alignedImg, warpMatrix, img1.size(),
                   cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);

    // 显示结果
    cv::imshow("Image 1", img1);
    cv::imshow("Image 2", img2);
    cv::imshow("Aligned Image 2", alignedImg);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void matchSparseOptFlow(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat img1_gray, img2_gray;
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);
    // 定义特征点
    std::vector<cv::Point2f> points1, points2;

    // 在第一幅图像中检测特征点
    cv::goodFeaturesToTrack(img1_gray, points1, 500, 0.01, 10);

    // 计算光流
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(img1_gray, img2_gray, points1, points2, status, err);

    // 过滤掉未跟踪到的特征点
    size_t i, k;
    for (i = k = 0; i < points2.size(); i++) {
        if (!status[i]) continue;
        points1[k] = points1[i];
        points2[k++] = points2[i];
    }
    points1.resize(k);
    points2.resize(k);

    // 计算单应性矩阵
    cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC);

    // 应用单应性矩阵进行图像配准
    cv::Mat alignedImg;
    cv::warpPerspective(img2, alignedImg, H, img2.size());

    // 显示结果
    cv::imshow("Image 1", img1);
    cv::imshow("Image 2", img2);
    cv::imshow("Aligned Image 2", alignedImg);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: ./det_keypoints <engine_path> <img1> <img2>" << std::endl;
        return -1;
    }
    CenterEngine center(argv[1]);
    cv::Mat img1 = cv::imread(argv[2]);
    cv::Mat img2 = cv::imread(argv[3]);
    // auto results1 = center.detect(img1);
    // auto results2 = center.detect(img2);

    // imgalign::FeatureMatch sift(argv[2], argv[3], imgalign::FeatureMatch::ORB);
    // sift.match(results1, results2);

    // matchECC(img1, img2);

    matchSparseOptFlow(img1, img2);
}