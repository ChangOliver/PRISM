#include "gpu_optimized_app.hpp"

#include <cstdio>

int main(int argc, char** argv) {
    GpuOptimizedArgs args;
    if (!parse_gpu_optimized_args(argc, argv, args, /*detailRequired=*/false)) {
        std::fprintf(stderr,
                     "Usage: %s -in <video> -h <height> -w <width> "
                     "-size <screen_inches|-1> -d <viewing_distance|-1> "
                     "[-out <summary_csv>] [-detail <detail_csv>] "
                     "[-lut <path>] [-HDR]\n",
                     argv[0]);
        return 1;
    }
    return run_gpu_optimized_pipeline(args);
}
