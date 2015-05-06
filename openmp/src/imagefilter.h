#ifndef OPENMP_IMAGEFILTER_H
#define OPENMP_IMAGEFILTER_H

#define CLAMP(x, low, high) ({\
  __typeof__(x) __x = (x); \
  __typeof__(low) __low = (low);\
  __typeof__(high) __high = (high);\
  __x > __high ? __high : (__x < __low ? __low : __x);\
  })


extern double sharpen_kernel[3][3];
extern double gauss_kernel[7][7];

struct _pixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};
typedef struct _pixel pixel;

int sharpen(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel);

int apply_kernel_to_image(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel,
                          double *kernel, int kernel_size);

#endif //OPENMP_IMAGEFILTER_H
