FILE(GLOB DET_SRC det/*.cpp det/*.cu)

cuda_add_library(centernet SHARED ${DET_SRC})

target_link_libraries(centernet ${CUDA_LIBRARIES})
target_link_libraries(centernet ${OpenCV_LIBS})
target_link_libraries(centernet nvinfer nvonnxparser)

add_executable(detection detection.cpp)
target_link_libraries(detection centernet)

add_executable(onnx2trt onnx2trt.cpp common/calibrator.cpp)
target_link_libraries(onnx2trt ${CUDA_LIBRARIES})
target_link_libraries(onnx2trt ${OpenCV_LIBS})
target_link_libraries(onnx2trt nvinfer nvonnxparser)

FILE(GLOB TRACK_SRC track/*.cpp)
add_library(botsort SHARED ${TRACK_SRC})

add_executable(track track.cpp)
target_link_libraries(track centernet)
target_link_libraries(track botsort)

FILE(GLOB IMG_ALIGN_SRC imagealignment/*.cpp)
add_library(imgalign SHARED ${IMG_ALIGN_SRC})

add_executable(det_keypoints imagealignment.cpp)
target_link_libraries(det_keypoints imgalign)
target_link_libraries(det_keypoints centernet)

add_executable(move_det movedet.cpp)
target_link_libraries(move_det ${OpenCV_LIBS})