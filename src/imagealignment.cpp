#include <iostream>

#include "det/centernet.h"
#include "det/centerutils.h"
#include "imagealignment/featurematch.h"

using namespace centernet;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: ./det_keypoints <engine_path> <img1> <img2>" << std::endl;
        return -1;
    }
    CenterEngine center(argv[1]);
    cv::Mat img1 = cv::imread(argv[2]);
    cv::Mat img2 = cv::imread(argv[3]);
    auto results1 = center.detect(img1);
    auto results2 = center.detect(img2);

    imgalign::FeatureMatch sift(argv[2], argv[3], imgalign::FeatureMatch::ORB);
    sift.match(results1, results2);
}