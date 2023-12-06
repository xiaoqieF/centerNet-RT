#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "det/centerutils.h"

void onnx2trt(const std::string& onnx_model,
              const std::string& save_engine_path,
              int build_type = 1) {
    using namespace nvinfer1;
    centernet::util::Logger logger;
    std::unique_ptr<IBuilder> builder{createInferBuilder(logger)};

    std::unique_ptr<INetworkDefinition> network{builder->createNetworkV2(
        1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH))};

    std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};
    parser->parseFromFile(onnx_model.c_str(), 1);
    for (int i = 0; i < parser->getNbErrors(); ++i) {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    std::cout << "TensorRT load onnx model sucessfully" << std::endl;
    std::unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    if (build_type == 1) {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (build_type == 2) {
        config->setFlag(BuilderFlag::kINT8);
    }

    std::unique_ptr<IHostMemory> model_stream{builder->buildSerializedNetwork(*network, *config)};
    std::cout << "Try to save engine now" << std::endl;
    std::ofstream p(save_engine_path, std::ios::binary);
    if (!p) {
        std::cerr << "could not open file: " << save_engine_path << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());

    std::cout << "Convert onnx model to TensorRT engine model successfully!" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: trans <onnx_model_path> <save_path>" << std::endl;
        return -1;
    }
    onnx2trt(argv[1], argv[2]);
    return 0;
}
