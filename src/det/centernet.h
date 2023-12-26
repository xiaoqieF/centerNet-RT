#ifndef CENTERRT_DET_CENTERNET_H
#define CENTERRT_DET_CENTERNET_H

#include <NvInfer.h>

#include <memory>
#include <string>
#include <vector>

#include "centerutils.h"

namespace centernet {
class CenterEngine {
public:
    CenterEngine(const std::string &engine_file);
    ~CenterEngine() {
        cudaStreamSynchronize(cuda_stream_);
        cudaStreamDestroy(cuda_stream_);
        for (auto &item : cuda_buffers_) {
            cudaFree(item);
        }
        cudaFree(cuda_output_buffer_);
    }

    std::vector<common::Detection> detect(const cv::Mat &img);
    void infer(const void *in_data, void *out_data);
    int64_t outputBufferSize() const { return output_buffer_size_; }

private:
    void initEngine();
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::vector<void *> cuda_buffers_;
    std::vector<int64_t> bind_buffer_sizes_;
    void *cuda_output_buffer_;
    int64_t output_buffer_size_;

    cudaStream_t cuda_stream_;
};

} // namespace centernet

#endif // CENTERRT_CENTERNET_H