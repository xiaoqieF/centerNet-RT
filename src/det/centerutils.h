#ifndef CENTERRT_CENTERUTIL_H
#define CENTERRT_CENTERUTIL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace util {

struct Box {
    float x1;
    float y1;
    float x2;
    float y2;
};

struct Detection {
    Box box;
    int class_id;
    float prob;
};

inline void CUDA_CHECK(cudaError_t error_code) {
    if (error_code != cudaSuccess) {
        std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;
        assert(0);
    }
}

inline int64_t volume(const nvinfer1::Dims& d) {
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        default:
            throw std::runtime_error("Invalid DataType.");
    }
    return 0;
}

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        using namespace std;
        string s;
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                s = "INTERNAL_ERROR";
                break;
            case Severity::kERROR:
                s = "ERROR";
                break;
            case Severity::kWARNING:
                s = "WARNING";
                break;
            case Severity::kINFO:
                s = "INFO";
                break;
            case Severity::kVERBOSE:
                s = "VERBOSE";
                break;
        }
        cerr << s << ": " << msg << endl;
    }
};

inline void* safeCudaMalloc(size_t memSize) {
    void* deviceMem;
    CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr) {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

void correctBox(std::vector<Detection>& results, const int img_w, const int img_h);
void drawImg(const std::vector<Detection>& results,
             cv::Mat& img,
             const std::vector<cv::Scalar>& color);

} // namespace util

#endif // CENTERRT_UTIL_H