cmake_minimum_required(VERSION 3.22)
project(epilepsy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
#enable_language(CUDA)

add_executable(epicap main.cpp utils.cpp)

#find_library(OpenCV)
#find_package(OpenCV REQUIRED PATHS ../build)
find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(epicap ${OpenCV_LIBS})
