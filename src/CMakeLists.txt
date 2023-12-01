FILE(GLOB DET_SRC det/*.cpp det/*.cu)

cuda_add_library(det SHARED ${DET_SRC})

target_link_libraries(det ${CUDA_LIBRARIES})
target_link_libraries(det ${OpenCV_LIBS})
target_link_libraries(det nvinfer nvonnxparser)

add_executable(infer infer.cpp)
target_link_libraries(infer det)

add_executable(trans onnx2trt.cpp)
target_link_libraries(trans ${CUDA_LIBRARIES})
target_link_libraries(trans ${OpenCV_LIBS})
target_link_libraries(trans nvinfer nvonnxparser)