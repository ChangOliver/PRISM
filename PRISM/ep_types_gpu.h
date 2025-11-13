#ifndef EP_TYPES_H
#define EP_TYPES_H

// CPU/GPU shared struct
typedef struct {
    float luminance;
    float red_ratio;
    float chroma_x;
    float chroma_y;
    bool isIncLum;
    bool isDecLum;
    bool isIncCol;
    bool isDecCol;
} epPixel;

#endif // EP_TYPES_H
