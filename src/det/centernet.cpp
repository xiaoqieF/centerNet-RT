#include "centernet.h"

#include <cassert>
#include <exception>
#include <fstream>
#include <iostream>

#include "config.h"
#include "postprocess.h"

namespace centernet {
static util::Logger logger;

namespace detail {
std::vector<char> readEngineFile(const std::string& engine_file) {
    // 从 engine 文件中读取数据
    std::vector<char> engine_data;
    std::ifstream f(engine_file, std::ios::binary);
    if (f.good()) {
        f.seekg(0, f.end);
        size_t size = f.tellg();
        f.seekg(0, f.beg);
        engine_data.resize(size);
        f.read(engine_data.data(), size);
        f.close();
    } else {
        std::cerr << "Could not read from: " << engine_file << std::endl;
        throw std::runtime_error("Read file failed");
    }
    return engine_data;
}
} // namespace detail

CenterEngine::CenterEngine(const std::string& engine_file) {
    using namespace nvinfer1;
    auto engine_data = detail::readEngineFile(engine_file);
    runtime_.reset(createInferRuntime(logger));
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    context_.reset(engine_->createExecutionContext());
    initEngine();
}

void CenterEngine::initEngine() {
    using namespace nvinfer1;
    int nb_bindings = engine_->getNbBindings(); // nb_bindings = 4
    std::cout << "Deserialize model successfully, Total bindings: " << engine_->getNbBindings()
              << std::endl;

    cuda_buffers_.resize(nb_bindings);
    bind_buffer_sizes_.resize(nb_bindings);
    const int max_batch_size = 1;
    int64_t total_size = 0;
    for (int i = 0; i < nb_bindings; ++i) {
        Dims dims = engine_->getBindingDimensions(i);
        DataType type = engine_->getBindingDataType(i);
        total_size = util::volume(dims) * max_batch_size * util::getElementSize(type);
        bind_buffer_sizes_[i] = total_size;
        cuda_buffers_[i] = util::safeCudaMalloc(total_size);
    }
    // 用于存放最终计算出的检测结果
    output_buffer_size_ = bind_buffer_sizes_[1] * 6;
    cuda_output_buffer_ = util::safeCudaMalloc(output_buffer_size_);
    util::CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
}

void CenterEngine::infer(const void* in_data, void* out_data) {
    const int max_batch_size = 1;
    int input_index = 0;
    util::CUDA_CHECK(cudaMemcpyAsync(cuda_buffers_[input_index], in_data,
                                     bind_buffer_sizes_[input_index], cudaMemcpyHostToDevice,
                                     cuda_stream_));
    context_->executeV2(&cuda_buffers_[input_index]);
    util::CUDA_CHECK(cudaMemset(cuda_output_buffer_, 0, sizeof(float)));
    centerNetPostProcess(
        static_cast<const float*>(cuda_buffers_[1]), static_cast<const float*>(cuda_buffers_[2]),
        static_cast<const float*>(cuda_buffers_[3]), static_cast<float*>(cuda_output_buffer_),
        config::input_w / 4, config::input_h / 4, config::class_num, config::kernel_size,
        config::vis_thresh);
    util::CUDA_CHECK(cudaMemcpyAsync(out_data, cuda_output_buffer_, output_buffer_size_,
                                     cudaMemcpyDeviceToHost, cuda_stream_));
}

} // namespace centernet
