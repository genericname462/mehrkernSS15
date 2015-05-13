#ifndef OPENMP_IMAGEFILTER_H
#define OPENMP_IMAGEFILTER_H

#define CLAMP(x, low, high) ({\
  __typeof__(x) __x = (x); \
  __typeof__(low) __low = (low);\
  __typeof__(high) __high = (high);\
  __x > __high ? __high : (__x < __low ? __low : __x);\
  })

#define POW_PI 9.869604401089358

extern double sharpen_kernel[3][3];
extern double gauss_kernel[7][7];
extern double soft_sharpen_kernel[3][3];

struct _pixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};
typedef struct _pixel pixel;

double lanczos(double x, int a);

double *generate_kernel(int range, double strength, int type);
/* type: 0 - sharpen
 *       1 - blur
 */

static double normal_dist(double x, double sigma) {
    return (1.0/sqrt(2 * M_PI * pow(sigma,1))) * pow(M_E, -(pow(x,2)/2*pow(sigma,2)));
}

int sharpen(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel);

int apply_kernel_to_image(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel,
                          double *kernel, int kernel_size);

int upscale_bilinear(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel, double factor);

int upscale_nearest_neightbour(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel, double factor);

int upscale_lanczos(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel, double factor);

int upscale_lanczos2(unsigned char *input, unsigned char *output, int x, int y, int n, int parallel, double factor);

#endif //OPENMP_IMAGEFILTER_H
