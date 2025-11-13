// For CCS25 Modifications

#include <iostream>
#include <cmath>
#include <vector>
#include <string_view>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <array>
#include <fstream>
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/intrin.hpp>

#include <chrono>

#include "utils.h"

using namespace std;
using namespace std::chrono;
using namespace cv;

#define NTHREADS    24
#define pixelDensity(w, h, s) sqrt(w*w + h*h) / s
#define minSafeArea(vd, pd)  vd*vd*pd*pd*0.1745*0.1309*0.25   // Technique G176: Keeping the flashing area small enough

struct Frame {
    Mat I;
    Mat Rr;
    Mat_<Point2f> S;
    int harmfulLumCount, harmfulColCount;
    Mat_<bool> isIncLum, isDecLum;
    Mat_<bool> isIncCol, isDecCol;

    Frame(int resolution_h, int resolution_w){
        I = Mat(resolution_h, resolution_w, CV_32F);
        Rr = Mat(resolution_h, resolution_w, CV_32F);
        S = Mat_<Point2f>(resolution_h, resolution_w, Point2f(0.0f, 0.0f));
        harmfulLumCount = harmfulColCount = 0;
        isIncLum = Mat_<bool>(resolution_h, resolution_w, false);
        isDecLum = Mat_<bool>(resolution_h, resolution_w, false);
        isIncCol = Mat_<bool>(resolution_h, resolution_w, false);
        isDecCol = Mat_<bool>(resolution_h, resolution_w, false);
    };

    Frame(){}; // default constructor
};
struct ThreadData {
    int index;
    int countLum, countCol;
};


Mat frame;
Frame f[2];
int resolution_h, resolution_w;
bool HDR = false;
constexpr int LUT_SIZE = 256;
vector<float> gammaLUT;

/* Fast inverse gamma correction using the lookup table */
inline float inverseGammaFast(float value) {

    if (value < 0)              return gammaLUT[0];
    else if (value >= LUT_SIZE) return gammaLUT[LUT_SIZE - 1];
    return gammaLUT[(int)value];
}

/* Calculate luminance and color of pixels */
void* calcLumColor(void* arg) {

    ThreadData* threadData = static_cast<ThreadData*>(arg);
    int index = threadData->index; // thread number
    int rowsPerThread = ceil(static_cast<double>(resolution_h) / NTHREADS);
    int startRow = index * rowsPerThread;
    int endRow = min(startRow + rowsPerThread, resolution_h);

    /* Process row group */
    for (int i = startRow; i < endRow; i++) {

        /* Set pointer position */
        Vec3f *pixelRow = frame.ptr<Vec3f>(i);
        auto *I_Row = f[1].I.ptr<float>(i);
        auto *Rr_Row = f[1].Rr.ptr<float>(i);
        auto *S_Row = f[1].S.ptr<Point2f>(i);

        /* Process pixels */
        for (int j = 0; j < resolution_w; j++) {

            Vec3f &pixel = pixelRow[j];

            /* Inverse Gamma Correction */
            float r = inverseGammaFast(pixel[2]);
            float g = inverseGammaFast(pixel[1]);
            float b = inverseGammaFast(pixel[0]);

            /* CIE XYZ color space conversion */
            float X = b * 0.1804375 + g * 0.3575761 + r * 0.4124564;
            float Y = b * 0.0721750 + g * 0.7151522 + r * 0.2126729;
            float Z = b * 0.9503041 + g * 0.1191920 + r * 0.0193339;

            /* Get chromaticity coordinates */
            S_Row[j].x = (X+Y+Z==0)? 0 : (4 * X) / (X + 15 * Y + 3 * Z);
            S_Row[j].y = (X+Y+Z==0)? 0 : (9 * Y) / (X + 15 * Y + 3 * Z);

            I_Row[j] = Y; // luminance calculation
            Rr_Row[j] = (r+g+b==0)? 0 : r / (r + g + b); // red ratio calculation
        }
    }

    return nullptr;
}

void* checkLumColThresh(void* arg) {

    ThreadData* threadData = static_cast<ThreadData*>(arg);
    int index = threadData->index; // thread number
    int rowsPerThread = ceil(static_cast<double>(resolution_h) / NTHREADS);
    int startRow = index * rowsPerThread;
    int endRow = min(startRow + rowsPerThread, resolution_h);

    int countLum = 0, countCol = 0;

    for (int i = startRow; i < endRow; i++){

        auto *frame_Row = frame.ptr<Vec3f>(i);

        auto *I1_Row = f[0].I.ptr<float>(i);
        auto *Rr1_Row = f[0].Rr.ptr<float>(i);
        auto *S1_Row = f[0].S.ptr<Point2f>(i);
        auto *I2_Row = f[1].I.ptr<float>(i);
        auto *Rr2_Row = f[1].Rr.ptr<float>(i);
        auto *S2_Row = f[1].S.ptr<Point2f>(i);

        auto *isIncLum1_Row = f[0].isIncLum.ptr<bool>(i);
        auto *isIncCol1_Row = f[0].isIncCol.ptr<bool>(i);
        auto *isDecLum1_Row = f[0].isDecLum.ptr<bool>(i);
        auto *isDecCol1_Row = f[0].isDecCol.ptr<bool>(i);
        auto *isIncLum2_Row = f[1].isIncLum.ptr<bool>(i);
        auto *isIncCol2_Row = f[1].isIncCol.ptr<bool>(i);
        auto *isDecLum2_Row = f[1].isDecLum.ptr<bool>(i);
        auto *isDecCol2_Row = f[1].isDecCol.ptr<bool>(i);

        for (int j = 0; j < resolution_w; j++) {
            float dx = S1_Row[j].x - S2_Row[j].x;
            float dy = S1_Row[j].y - S2_Row[j].y; 

            bool isHarmfulLum = (abs(I1_Row[j]-I2_Row[j]) > 0.1) && (I1_Row[j] < 0.8 || I2_Row[j] < 0.8);
            bool isHarmfulCol = (Rr1_Row[j] >= 0.8 || Rr2_Row[j] >= 0.8) && ((dx*dx + dy*dy) > 0.04); // avoid sqrt for speed

            /* Michaelson contrast for HDR*/
            if (HDR){
                isHarmfulLum = isHarmfulLum || ((abs(I1_Row[j]-I2_Row[j])/(I1_Row[j]+I2_Row[j]) > 0.05882352941) && (I1_Row[j] > 0.8 && I2_Row[j] > 0.8)); // 0.0588235294118
            }    

            if (isHarmfulLum){
                isIncLum2_Row[j] = (I1_Row[j] < I2_Row[j]);
                isDecLum2_Row[j] = (I1_Row[j] > I2_Row[j]);
                if ( (isIncLum1_Row[j] && isDecLum2_Row[j]) || (isDecLum1_Row[j] && isIncLum2_Row[j]) ){
                    countLum += 1;
                    isDecLum2_Row[j] = false;
                    isIncLum2_Row[j] = false;
                }            
            }
            else{
                isIncLum2_Row[j] = isIncLum1_Row[j];
                isDecLum2_Row[j] = isDecLum1_Row[j];
            }
            if (isHarmfulCol){
                isIncCol2_Row[j] = (Rr1_Row[j] < Rr2_Row[j]);
                isDecCol2_Row[j] = (Rr1_Row[j] > Rr2_Row[j]);
                if ( (isIncCol1_Row[j] && isDecCol2_Row[j]) || (isDecCol1_Row[j] && isIncCol2_Row[j]) ){
                    countCol += 1;
                    isDecCol2_Row[j] = false;
                    isIncCol2_Row[j] = false;
                }             
            } 
            else{
                isIncCol2_Row[j] = isIncCol1_Row[j];
                isDecCol2_Row[j] = isDecCol1_Row[j];
            }
        }
    }

    threadData->countLum = countLum;
    threadData->countCol = countCol;

    return nullptr;
}

/* ./adaptive -in infile -out outfile [(-h xx -w xx -size xx -d xx)] [-HDR]*/
int main(int argc, char * argv[]) {

    /* Parse command line arguments */
    const std::vector<std::string_view> args(argv, argv + argc);
    
    /* Parse filename */
    const string filename = has_option(args, "-in")? get_option(args, "-in").data() : "";
    if (filename == ""){
        cout << "No file provided (Use the -in option)" << endl;
        return -1;
    }
    const string outfilename = has_option(args, "-out")? get_option(args, "-out").data() : "result.csv";

    int screenSize, viewingDistance;

    if (!has_option(args, "-h") || !has_option(args, "-w") || !has_option(args, "-size") || !has_option(args, "-d") ){
        cout << "Missing arguments" << endl;
        return -1;
    }
    resolution_h = atoi(get_option(args, "-h").data());
    resolution_w = atoi(get_option(args, "-w").data());
    screenSize = atoi(get_option(args, "-size").data());
    viewingDistance = atoi(get_option(args, "-d").data());
    double pixelDensity = (screenSize==-1)? 0 : pixelDensity(resolution_w, resolution_h, screenSize);
    int minSafeArea = (viewingDistance==-1)? 0.25*resolution_h*resolution_w : minSafeArea(viewingDistance, pixelDensity); 

    if (has_option(args, "-HDR")){
        HDR = true;
    }
    

    /* Prepare inverse gamma lookup table */
    const std::string lutCandidates[] = {
        "inverseGammaLUT.bin",
        "../inverseGammaLUT.bin"
    };
    bool lutLoaded = false;
    for (const auto& candidate : lutCandidates) {
        std::ifstream lutFile(candidate, std::ios::binary);
        if (lutFile.good()) {
            lutFile.close();
            gammaLUT = readBinaryFile(candidate);
            lutLoaded = true;
            break;
        }
    }
    if (!lutLoaded) {
        cout << "Could not locate inverseGammaLUT.bin" << endl;
        return -1;
    }

    auto start = high_resolution_clock::now();

    // /* Frame struct initilization */
    f[0] = Frame(resolution_h, resolution_w);
    f[1] = Frame(resolution_h, resolution_w);

    /* open video file */
    VideoCapture cap(filename);
    if (!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    /* get FPS */
    const int fps = cap.get(CAP_PROP_FPS);

    int freqLum = 0, freqCol = 0;

    queue<Frame> oneSecFrames;

    bool hasFlash = false, hasRed = false;
    pthread_t threads[NTHREADS];
    ThreadData threadData[NTHREADS];
    
    /* Main loop */
    while (cap.read(frame)) {

        frame.convertTo(frame, CV_32FC3);

        /* Multi-threaded luminance and color calculation */
        for (int i = 0; i < NTHREADS; i++){
            threadData[i].index = i;
            if (pthread_create(&threads[i], nullptr, &calcLumColor, &threadData[i]) != 0){
                perror("Failed to create thread");
            }
        }
        for (int i = 0; i < NTHREADS; i++){
            if (pthread_join(threads[i], nullptr) != 0) {
                perror("Failed to join thread");
            }
        }

        /* Multi-threaded harmful luminance and color differences check */
        for (int i = 0; i < NTHREADS; i++){
            threadData[i].index = i;
            if (pthread_create(&threads[i], nullptr, &checkLumColThresh, &threadData[i]) != 0){
                perror("Failed to create thread");
            }
        }

        for (int i = 0; i < NTHREADS; i++){
            if (pthread_join(threads[i], nullptr) != 0) {
                perror("Failed to join thread");
            }
        }

        for (int i = 0; i < NTHREADS; i++){
            f[1].harmfulLumCount += threadData[i].countLum;
            f[1].harmfulColCount += threadData[i].countCol;
        }

        /* Check area threshold && Update total number of harmful flashes */
        if (f[1].harmfulLumCount > minSafeArea) freqLum++;
        if (f[1].harmfulColCount > minSafeArea) freqCol++;        

        /* Sliding window keeps only one-second worth of frames */
        oneSecFrames.push(f[1]);

        if (oneSecFrames.size() > fps){
            if (oneSecFrames.front().harmfulLumCount > minSafeArea) freqLum--;
            if (oneSecFrames.front().harmfulColCount > minSafeArea) freqCol--;
            oneSecFrames.pop();
        } 

         /* Check frequency threshold */
        if (freqLum > 3) hasFlash = true;
        if (freqCol > 3) hasRed = true;

        if(hasFlash and hasRed){
            break;
        }
        
        /* Shift frames by 1 */
        swap(f[0], f[1]);
        f[1] = Frame(resolution_h, resolution_w);
    }

    
    // cout << "hasFlash " << hasFlash << endl;
    // cout << "hasRed " << hasRed << endl;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    const int frame_count = int(cap.get(CAP_PROP_FRAME_COUNT));
    const int vid_length = frame_count/fps;

    // Print runtime statistics
    cout << "Processed " << frame_count << " frames in " << duration.count() << " ms" << endl;

    FILE* outfile;
    outfile = fopen(outfilename.c_str(), "a");
    fprintf(outfile, "%s,%d,%d,%ld,%d,%d\n", basename(argv[2]), hasFlash, hasRed, duration.count(), vid_length, frame_count);
    fclose(outfile);
   
    return 0;
}
