cmake_minimum_required(VERSION 3.22)
project(epicap_gpu_optimized LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(gpu_optimized_core STATIC
    gpu_optimized_app.cu
)

set_target_properties(gpu_optimized_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(gpu_optimized_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(gpu_optimized_core
    PUBLIC
        ${OpenCV_LIBRARIES}
        CUDA::cudart
        CUDA::cuda_driver
        Threads::Threads
)

add_executable(epicap_gpu_optimized main_gpu_correct.cu)
target_link_libraries(epicap_gpu_optimized
    PRIVATE gpu_optimized_core
)
