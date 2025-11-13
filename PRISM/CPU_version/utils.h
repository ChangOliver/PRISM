#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <string_view>
#include <vector>

using namespace cv;
using namespace std;

std::string_view get_option(
    const std::vector<std::string_view>& args, 
    const std::string_view& option_name);

bool has_option(
    const std::vector<std::string_view>& args, 
    const std::string_view& option_name);

vector<float> readBinaryFile(string fileName);

// void writeBinaryFile(string fileName, vector<float> arr, size_t len);
// void createGammaLUT();

#endif