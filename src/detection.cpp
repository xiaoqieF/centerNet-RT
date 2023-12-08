#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <opencv2/imgproc/types_c.h>

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "common/datatype.h"
#include "det/centernet.h"
#include "det/centerutils.h"
#include "det/config.h"

using namespace centernet;

std::vector<float> prepareImage(cv::Mat& img) {
    int channel = config::channel;
    int input_w = config::input_w;
    int input_h = config::input_h;
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

void printGpuInfo() {
    int dev = 0;
    cudaDeviceProp devProp;
    util::CUDA_CHECK(cudaGetDeviceProperties(&devProp, dev));
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB"
              << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./infer <engine_path> <img_path>";
        return -1;
    }
    printGpuInfo();

    CenterEngine center(argv[1]);
    cv::Mat img = cv::imread(argv[2]);
    auto input = prepareImage(img);

    std::unique_ptr<float[]> output_data(new float[center.outputBufferSize()]);
    auto t0 = std::chrono::steady_clock::now();
    center.infer(input.data(), output_data.get());
    int num_det = static_cast<int>(output_data[0]);
    std::cout << "det_num: " << num_det << std::endl;
    std::vector<common::Detection> results(num_det);
    memcpy(results.data(), &output_data[1], num_det * sizeof(common::Detection));
    for (auto& det : results) {
        std::cout << "class id: " << det.class_id << "; prob: " << det.prob
                  << "; bbox: x1: " << det.box.x1 << " y1: " << det.box.y1 << " x2: " << det.box.x2
                  << " y2: " << det.box.y2 << std::endl;
    }
    util::correctBox(results, img.cols, img.rows);
    std::vector<cv::Scalar> color{cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0)};
    util::drawImg(results, img, color);
    auto t1 = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    std::cout << "Cost: " << dur.count() << " microseconds" << std::endl;
    cv::namedWindow("result", cv::WINDOW_NORMAL);
    cv::resizeWindow("result", 1024, 768);
    cv::imshow("result", img);
    cv::waitKey(0);
}