#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "imagefilter.h"


double sharpen_kernel[3][3] = {{-1,-1,-1},
                               {-1,+9,-1},
                               {-1,-1,-1}};

double gauss_kernel[7][7] = {
        {0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067},
        {0.00002292, 0.00078634, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292},
        {0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117},
        {0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771},
        {0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117},
        {0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292},
        {0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067}
};

int apply_kernel_to_image(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel,
                          double *kernel, int kernel_size) {
    memcpy(output, input, (size_t) x * y * n);
    double sum;
    int kernel_size_offset = (kernel_size-1)/2;
    #pragma omp parallel if(parallel) private(sum)
    #pragma omp for
    for (int j = kernel_size_offset; j < y-(kernel_size_offset); ++j) {
        for (int i = kernel_size_offset; i < x-(kernel_size_offset); ++i) {
            for (int c = 0; c < n; ++c) {
                // Compute kernel
                sum = 0;
                for (int ki = 0; ki < kernel_size; ++ki) {
                    for (int kj = 0; kj < kernel_size; ++kj) {
                        sum += kernel[kj * kernel_size + ki] * (signed int) input[(j+kj-kernel_size_offset) * x * n + (i+ki-kernel_size_offset) * n + c];
                    }
                }
                // Apply kernel
                output[(j * x * n) + (i * n) + c] = (unsigned char) CLAMP(sum, 0, 255);
            }
        }
    }
    return 0;
}

int sharpen(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel) {
    // Assumes that input and output are continuous blocks of data:
    // [Pixel 00|Pixel 01|Pixel 02|...|Pixel 0x|Pixel 10|Pixel 11|...|Pixel yx]
    // Pixel format on little endian: [ ALPHA | BLUE | GREEN | RED ]
    // MUST be 32 bit wide, 8 bit per component
    // uint32_t black = 0xff000000;
    // uint32_t white = 0xffffffff;
    // Or use the pixel struct:
    // (*pixel_ptr_to_image_in_memory)[y][x].r for the red value of the pixel at position x,y
    memcpy(output, input, (size_t) x * y * n);
    signed int sum;
    #pragma omp parallel if(parallel) shared(output) private(sum)
    #pragma omp for
    // Ignores edge cases where i-1 would be out of the picture aka negative or bigger x-1
    for (int j = 1; j < y-1; ++j) {
        for (int i = 1; i < x-1; ++i) {
            // Loop over the color layers
            for (int c = 0; c < n; ++c) {
                // Compute kernel
                sum = -1 * (signed int) input[(j -1) * x * n + (i -1) * n + c] + -1 * (signed int) input[(j -1) * x * n + (i) * n + c] + -1 * (signed int) input[(j -1) * x * n + (i +1) * n + c] +
                      -1 * (signed int) input[(j) * x * n + (i -1) * n + c] + 9 * (signed int) input[(j) * x * n + (i) * n + c] + -1 * (signed int) input[(j) * x * n + (i +1) * n + c] +
                      -1 * (signed int) input[(j +1) * x * n + (i -1) * n + c] + -1 * (signed int) input[(j +1) * x * n + (i) * n + c] + -1 * (signed int) input[(j +1) * x * n + (i +1) * n + c];

                // Apply kernel
                output[(j * x * n) + (i * n) + c] = (unsigned char) CLAMP(sum, 0, 255);
            }
        }
    }
    return 0;
}