#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "common/calibrator.h"
#include "det/centerutils.h"

void onnx2trt(const std::string& onnx_model,
              const std::string& save_engine_path,
              int build_type = 1,
              const std::string& image_dir = "",
              const char* calib_table = nullptr) {
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
        if (builder->platformHasFastFp16()) {
            std::cout << "Use Fast FP16 !!!" << std::endl;
            config->setFlag(BuilderFlag::kFP16);
        } else {
            std::cout << "Unsupport FP16 mode" << std::endl;
            exit(-1);
        }
    }
    if (build_type == 2) {
        if (builder->platformHasFastInt8()) {
            std::cout << "Use Fast Int8 !!!" << std::endl;
            config->setFlag(BuilderFlag::kINT8);
            if (image_dir == "" || calib_table == nullptr) {
                std::cout << "Please assign image_dir and calib_table" << std::endl;
                exit(-1);
            }
            nvinfer1::IInt8EntropyCalibrator2* calibrator =
                new Calibrator(1, 512, 512, image_dir, calib_table);
            config->setInt8Calibrator(calibrator);
        } else {
            std::cout << "Unsupport Int8 mode" << std::endl;
            exit(-1);
        }
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
    if (argc != 3 && argc != 5) {
        std::cerr << "Usage: trans <onnx_model_path> <save_path> optional<image_dir> "
                     "optional<calib_table>"
                  << std::endl;
        return -1;
    }
    if (argc == 3) {
        onnx2trt(argv[1], argv[2]);
    } else if (argc == 5) {
        std::string image_dir = argv[3];
        onnx2trt(argv[1], argv[2], 2, image_dir, argv[4]);
    }
    return 0;
}
