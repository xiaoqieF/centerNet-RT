cmake_minimum_required(VERSION 3.16)
project(centerrt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-g -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# opencv 
find_package(OpenCV)
message(STATUS " libraries: ${OpenCV_LIBS}")
message(STATUS " include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
message(STATUS " libraries: ${CUDA_LIBRARIES}")
message(STATUS " include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA) 

add_executable(trans onnx2trt.cpp)
add_executable(infer infer.cpp)
cuda_add_library(det SHARED src/centernet.cpp src/centerutil.cpp src/postprocess.cu)

target_link_libraries(infer det)

target_link_libraries(trans ${CUDA_LIBRARIES})
target_link_libraries(trans ${OpenCV_LIBS})
target_link_libraries(trans nvinfer nvonnxparser)

target_link_libraries(det ${CUDA_LIBRARIES})
target_link_libraries(det ${OpenCV_LIBS})
target_link_libraries(det nvinfer nvonnxparser)