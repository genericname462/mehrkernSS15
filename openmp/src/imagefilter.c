#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

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

double soft_sharpen_kernel[3][3] = {
        {-0.375, -0.375, -0.375},
        {-0.375,      4, -0.375},
        {-0.375, -0.375, -0.375}
};

double *generate_kernel(int range, double strength, int type) {
    int width = 1 + range * 2;
    double outer = (strength) / (width*width - 1);
    double *kernel = malloc(width * width * sizeof(double));
    if (type == 0) {
        for (int j = 0; j < width; ++j) {
            for (int i = 0; i < width; ++i) {
                kernel[j * width + i] = -outer;
            }
        }
        kernel[range * width + range] = strength+1;
    } else if (type == 1) {
        for (int j = 0; j < width; ++j) {
            for (int i = 0; i < width; ++i) {
                kernel[j * width + i] = normal_dist(range - j, strength) * normal_dist(range - i, strength);
            }
        }
    }
    return kernel;
}

double lanczos(double x, int a) {
    if (x != 0)
        return ((a * sin(M_PI * x) * sin(M_PI * x/a))/(POW_PI * x * x));
    return 1;

    /*if (0 < fabs(x) && fabs(x) < a) {
        return ((a * sin(M_PI * x) * sin(M_PI * x/a))/(POW_PI * x * x));
    } else {
        return 0;
    }*/
}

int apply_kernel_to_image(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel,
                          double *kernel, int kernel_size) {
    memcpy(output, input, (size_t) x * y * n);
    double sum;
    int kernel_size_offset = (kernel_size-1)/2;
    #pragma omp parallel if(parallel) private(sum)
    {
        #pragma omp for
        for (int j = kernel_size_offset; j < y - (kernel_size_offset); ++j) {
            for (int i = kernel_size_offset; i < x - (kernel_size_offset); ++i) {
                for (int c = 0; c < n; ++c) {
                    // Compute kernel
                    sum = 0;
                    for (int ki = 0; ki < kernel_size; ++ki) {
                        for (int kj = 0; kj < kernel_size; ++kj) {
                            sum += kernel[kj * kernel_size + ki] *
                                   (double) input[(j + kj - kernel_size_offset) * x * n +
                                                  (i + ki - kernel_size_offset) * n + c];
                        }
                    }
                    // Apply kernel
                    output[(j * x * n) + (i * n) + c] = (unsigned char) CLAMP(sum, 0, 255);
                }
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

int upscale_bilinear(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel, double factor) {
    int x_out = (int) round(x*factor);
    int y_out = (int) round(y*factor);

    int x_0;
    int x_1;
    int y_0;
    int y_1;

    int f_y_0_x_0;
    int f_y_0_x_1;
    int f_y_1_x_0;
    int f_y_1_x_1;

    double f_y_0_inter;
    double f_y_1_inter;

    signed int sum;
    printf("max: %i,%i\n", x_out, y_out);

    factor += 0.0001;

    for (int j = 0; j < y_out; ++j) {
        for (int i = 0; i < x_out; ++i) {
            x_0 = (int)floor((i*1.)/factor);
            x_1 = (int)ceil((i*1.)/factor);
            if (x_1 == x) {
                --x_1;
            }
            y_0 = (int)floor((j*1.)/factor);
            y_1 = (int)ceil((j*1.)/factor);
            printf("working on %i,%i with remote (%i to %i, %i to %i)\n", i, j, x_0, x_1, y_0, y_1);
            for (int c = 0; c < n; ++c) {
                f_y_0_x_0 = input[y_0 * x*n + x_0*n + c];
                f_y_0_x_1 = input[y_0 * x*n + x_1*n + c];
                f_y_1_x_0 = input[y_1 * x*n + x_0*n + c];
                f_y_1_x_1 = input[y_1 * x*n + x_1*n + c];

//                f_y_0_inter = f_y_0_x_0 + ((f_y_0_x_1 - f_y_0_x_0) / (x_1 - x_0)) * ((1.*i)/factor - x_0);
//                f_y_1_inter = f_y_1_x_0 + ((f_y_1_x_1 - f_y_1_x_0) / (x_1 - x_0)) * ((1.*i)/factor - x_0);
//
//                sum = f_y_0_inter + ((f_y_1_inter - f_y_0_inter) / (y_1 - y_0)) * ((1.*j)/factor - y_0);

                if (x_0 != x_1) {
                    f_y_0_inter = f_y_0_x_0 + (f_y_0_x_1 - f_y_0_x_0) * ((i/factor - x_0)/(x_1 - x_0));
                    f_y_1_inter = f_y_1_x_0 + (f_y_1_x_1 - f_y_1_x_0) * ((i/factor - x_0)/(x_1 - x_0));
                } else {
                    f_y_0_inter = f_y_0_x_0;
                    f_y_1_inter = f_y_1_x_0;
                }
                if (y_0 != y_1) {
                    sum = f_y_0_inter + (f_y_1_inter - f_y_0_inter) * ((j/factor - y_0) / (y_1 - y_0));
                }  else {
                    sum = f_y_0_inter;
                }

                output[((j) * x_out*n + i*n + c)] = (unsigned char) sum;
            }
        }
    }


//    unsigned char *bak = malloc((size_t) x*2 * y*2 * n);
//    memcpy(bak, output, (size_t) x*2 * y*2 * n);
//    double *mykernel = generate_kernel(3,0.7,0);
//
//    apply_kernel_to_image(bak, output, x*2, y*2, n, 1, mykernel, 7);
//
//    free(mykernel);
//    free(bak);



//    printf("end: %f,%f", x_org, y_org);
    return 0;
}


int upscale_lanczos(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel, double factor) {
    int x_out = (int) round(x*factor);
    int y_out = (int) round(y*factor);

    int a = 3;

    #pragma omp parallel if(parallel) shared(x_out, y_out, output, input, factor, a)
    {
        double sum;
        double omega;
        double lan;
        double f_i;
        double f_j;
        #pragma omp for
        for (int j = a+1; j < y_out - a; ++j) {
            for (int i = a+1; i < x_out - a; ++i) {
                for (int c = 0; c < n; ++c) {
                    sum = 0;
                    omega = 0;
                    for (int k = -a + 1; k <= a; ++k) {
                        for (int l = -a + 1; l <= a; ++l) {
                            f_i = floor((1. * i)/ factor);
                            f_j = floor((1. * j)/ factor);
                            lan = lanczos(k - ((1. * i) / factor) + f_i, a) *
                                  lanczos(l - ((1. * j) / factor) + f_j, a);
                            //lan = 0;
                            sum += (double) input[(int) ((f_j + l) * x * n +
                                                         (f_i + k) * n + c)] *
                                    lan;
                            omega += lan;
                        }
                    }
                    output[(j * x_out * n) + (i * n) + c] = CLAMP(sum / omega, 0, 255);
                }
            }
        }
    }
    return 0;
}

int upscale_lanczos2(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel, double factor) {
    int x_out = (int) round(x*factor);
    int y_out = (int) round(y*factor);

    int a = 3;
    unsigned char *temp = malloc((size_t) (x*factor * y*factor * n));

    #pragma omp parallel if(parallel) shared(x_out, y_out, output, input, factor, a)
    {
        double sum;
        double omega;
        double lan;
        double f_i;
        double f_j;
        #pragma omp for
        for (int j = a+1; j < y_out - a; ++j) {
            for (int i = a+1; i < x_out - a; ++i) {
                for (int c = 0; c < n; ++c) {
                    sum = 0;
                    omega = 0;
                    for (int k = -a + 1; k <= a; ++k) {
                        f_i = floor((1. * i)/ factor);
                        f_j = floor((1. * j)/ factor);
                        lan = lanczos(k - ((1. * i) / factor) + f_i, a);
                        sum += (double) input[(int) ((f_j) * x * n +
                                                     (f_i + k) * n + c)] *
                               lan;
                        omega += lan;
                    }
                    temp[(j * x_out * n) + (i * n) + c] = CLAMP(sum / omega, 0, 255);
                }
            }
        }
        //memcpy(temp, input, (size_t) (x*factor * y*factor * n));

        for (int j = a+1; j < y_out - a; ++j) {
            for (int i = a+1; i < x_out - a; ++i) {
                for (int c = 0; c < n; ++c) {
                    sum = 0;
                    omega = 0;
                    for (int k = -a + 1; k <= a; ++k) {
                        lan = lanczos(k - ((1. * j) / factor) + j, a);
                        sum += (double) temp[((j) * x * n + i * n + c)] * lan;
                        omega += lan;
                    }
                    output[(j * x_out * n) + (i * n) + c] = CLAMP(sum / omega, 0, 255);
                }
            }
        }
    }
    free(temp);
    return 0;
}

int upscale_nearest_neightbour(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel,
                               double factor) {
    int x_out = x*factor;
    int y_out = y*factor;
    for (int j = 0; j < y_out; ++j) {
        for (int i = 0; i < x_out; ++i) {
            for (int c = 0; c < n; ++c) {
                output[j * x_out*n + i*n + c] = input[(int)floor(j/factor) * x*n + (int)floor(i/factor)*n + c];
            }
        }
    }
    return 0;
}
