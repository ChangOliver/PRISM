#ifndef EPICAP_GPU_OPTIMIZED_APP_HPP
#define EPICAP_GPU_OPTIMIZED_APP_HPP

#include <string>

struct GpuOptimizedArgs {
    std::string input_path;
    std::string output_csv = "result.csv";
    std::string detail_csv;
    std::string lut_path;
    int width = 0;
    int height = 0;
    int screen_size = -1;
    int viewing_distance = -1;
    bool hdr = false;
    bool profile = false;
};

bool parse_gpu_optimized_args(int argc, char** argv, GpuOptimizedArgs& out, bool detailRequired);
int run_gpu_optimized_pipeline(const GpuOptimizedArgs& cfg);

#endif // EPICAP_GPU_OPTIMIZED_APP_HPP
