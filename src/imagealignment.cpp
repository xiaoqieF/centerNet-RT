#include <iostream>

#include "det/centernet.h"
#include "det/centerutils.h"
#include "imagealignment/featurematch.h"

using namespace centernet;

std::vector<common::Detection> detect(CenterEngine& engine, const cv::Mat& img) {
    auto input = util::prepareImage(img);

    std::unique_ptr<float[]> output_data(new float[engine.outputBufferSize()]);
    auto t0 = std::chrono::steady_clock::now();
    engine.infer(input.data(), output_data.get());
    int num_det = static_cast<int>(output_data[0]);
    std::cout << "det_num: " << num_det << std::endl;
    std::vector<common::Detection> results(num_det);
    memcpy(results.data(), &output_data[1], num_det * sizeof(common::Detection));
    util::correctBox(results, img.cols, img.rows);
    return results;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: ./det_keypoints <engine_path> <img1> <img2>" << std::endl;
        return -1;
    }
    CenterEngine center(argv[1]);
    cv::Mat img1 = cv::imread(argv[2]);
    cv::Mat img2 = cv::imread(argv[3]);
    auto results1 = detect(center, img1);
    auto results2 = detect(center, img2);

    imgalign::FeatureMatch sift(argv[2], argv[3], imgalign::FeatureMatch::ORB);
    sift.match(results1, results2);
}