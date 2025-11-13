#include <string_view>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
// #include <opencv2/cudawarping.hpp>

#include "utils.h"

using namespace cv;
using namespace cv::cuda;
using namespace std;

std::string_view get_option(
    const std::vector<std::string_view>& args, 
    const std::string_view& option_name) {
    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == option_name)
            if (it + 1 != end)
                return *(it + 1);
    }
    
    return "";
}

bool has_option(
    const std::vector<std::string_view>& args, 
    const std::string_view& option_name) {
    for (auto it = args.begin(), end = args.end(); it != end; ++it) {
        if (*it == option_name)
            return true;
    }
    
    return false;
}

vector<float> readBinaryFile(string fileName) {
    vector<float> ret;
    float f;
    ifstream in(fileName, std::ios::binary);
    while (in.read(reinterpret_cast<char*>(&f), sizeof(float))) {
        ret.push_back(f);
    }
    return ret;
 }

// void writeBinaryFile(string fileName, vector<float> arr, size_t len) {
//     ofstream f(fileName, std::ios::out | std::ios::binary);
//     f.write((char*)arr.data(), arr.size() * sizeof( decltype(arr)::value_type ));
// }

/* Create the inverse gamma lookup table*/
// void createGammaLUT() {
//     vector<float> gammaLUT;
//     for (int i = 0; i < 256; i++) {
//         float x = static_cast<float>(i) / 255.0f;
//         if (x <= 0.04045) {
//             gammaLUT.push_back(x / 12.92);
//         } else {
//             gammaLUT.push_back(pow((x + 0.055) / 1.055, 2.4));
//         }
//     }
//     writeBinaryFile("inverseGammaLUT.bin", gammaLUT, 256);
// }