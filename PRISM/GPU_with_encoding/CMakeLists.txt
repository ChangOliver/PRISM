# CMake project for the in-encoder GPU runtime.

cmake_minimum_required(VERSION 3.22)
project(epicap_gpu_runtime LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(epicap_gpu_runtime STATIC
    src/runtime.cpp
    src/gpu_kernels.cu
)

target_include_directories(epicap_gpu_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(epicap_gpu_runtime
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
)

set_target_properties(epicap_gpu_runtime PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

find_path(FFMPEG_INCLUDE_DIR
    NAMES libavcodec/avcodec.h
    PATHS /usr/include /usr/local/include /usr/include/x86_64-linux-gnu
          /usr/local/include/x86_64-linux-gnu
)
if (NOT FFMPEG_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find FFmpeg headers (libavcodec/avcodec.h)")
endif()

find_library(AVCODEC_LIB avcodec)
find_library(AVUTIL_LIB avutil)
find_library(SWSCALE_LIB swscale)
if (NOT AVCODEC_LIB OR NOT AVUTIL_LIB OR NOT SWSCALE_LIB)
    message(FATAL_ERROR "Failed to find required FFmpeg libraries (avcodec, avutil, swscale)")
endif()

add_executable(epicap_runtime_pipeline
    runtime_pipeline.cpp
    src/recon_encoder.cpp
)
target_link_libraries(epicap_runtime_pipeline
    PRIVATE
        epicap_gpu_runtime
        ${OpenCV_LIBRARIES}
        ${AVCODEC_LIB}
        ${AVUTIL_LIB}
        ${SWSCALE_LIB}
)
target_include_directories(epicap_runtime_pipeline
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${FFMPEG_INCLUDE_DIR}
)
