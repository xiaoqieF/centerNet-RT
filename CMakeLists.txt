cmake_minimum_required(VERSION 3.16)
project(centerrt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# opencv 
find_package(OpenCV REQUIRED)
message(STATUS " libraries: ${OpenCV_LIBS}")
message(STATUS " include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
message(STATUS " libraries: ${CUDA_LIBRARIES}")
message(STATUS " include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA) 

find_package(Eigen3 REQUIRED)
message(STATUS " include path: ${EIGEN3_INCLUDE_DIRS}")

include_directories(src)
add_subdirectory(src)