#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "det/centernet.h"
#include "det/centerutils.h"
#include "det/config.h"

std::vector<float> prepareImage(cv::Mat& img) {
    int channel = centernet::config::channel;
    int input_w = centernet::config::input_w;
    int input_h = centernet::config::input_h;
    float scale = cv::min(float(input_w) / img.cols, float(input_h) / img.rows);
    auto scale_size = cv::Size(img.cols * scale, img.rows * scale);
    cv::Mat resized;
    cv::resize(img, resized, scale_size, 0, 0);

    cv::Mat cropped(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Rect rect((input_w - scale_size.width) / 2, (input_h - scale_size.height) / 2,
                  scale_size.width, scale_size.height);
    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    cropped.convertTo(img_float, CV_32FC3, 1. / 255);

    std::vector<float> res(input_h * input_w * channel);
    auto data = res.data();
    int channel_len = input_h * input_w;
    // HWC -> CHW
    std::vector<cv::Mat> input_channels{
        cv::Mat(input_h, input_w, CV_32FC1, data + channel_len * 2),
        cv::Mat(input_h, input_w, CV_32FC1, data + channel_len * 1),
        cv::Mat(input_h, input_w, CV_32FC1, data + channel_len * 0)};
    cv::split(img_float, input_channels);
    return res;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./track <video path>" << std::endl;
        return -1;
    }
    centernet::CenterEngine engine("../onnxmodel/DroneVsBirds_centernetplus_r18.trt");
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cout << "Can't open video " << argv[1] << std::endl;
        return -1;
    }
    std::cout << "视频中图像的宽度=" << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "视频中图像的高度=" << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "视频帧率=" << cap.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "视频的总帧数=" << cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::unique_ptr<float[]> output_data(new float[engine.outputBufferSize()]);
    cv::Mat img;
    while (true) {
        cap >> img;
        if (img.empty()) {
            break;
        }
        auto t0 = std::chrono::steady_clock::now();
        auto net_input = prepareImage(img);
        engine.infer(net_input.data(), output_data.get());
        int num_det = static_cast<int>(output_data[0]);
        // std::cout << "det_num: " << num_det << std::endl;
        std::vector<centernet::util::Detection> results(num_det);
        memcpy(results.data(), &output_data[1], num_det * sizeof(centernet::util::Detection));
        // for (auto& det : results) {
        //     std::cout << "class id: " << det.class_id << "; prob: " << det.prob
        //               << "; bbox: x1: " << det.box.x1 << " y1: " << det.box.y1
        //               << " x2: " << det.box.x2 << " y2: " << det.box.y2 << std::endl;
        // }
        centernet::util::correctBox(results, img.cols, img.rows);
        std::vector<cv::Scalar> color{cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0)};
        centernet::util::drawImg(results, img, color);
        auto t1 = std::chrono::steady_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
        // std::cout << "Cost: " << dur.count() << " microseconds" << std::endl;
    }

    return 0;
}