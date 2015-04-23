#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "../lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#define CLAMP(x, low, high) ({\
  __typeof__(x) __x = (x); \
  __typeof__(low) __low = (low);\
  __typeof__(high) __high = (high);\
  __x > __high ? __high : (__x < __low ? __low : __x);\
  })

int print_matrix(double *matrix, size_t n, size_t m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            printf("%f\t", matrix[i*n + j]);
        }
        printf("\n");
    }
    return 0;
}

int print_vector(double *vector, size_t m) {
    for (int i = 0; i < m; ++i) {
        printf("%f ", vector[i]);
        printf("\n");
    }
    return 0;
}

int scale_matrix(double *matrix, size_t n, size_t m, int scale) {
    #pragma omp parallel for shared(matrix,n,m)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            matrix[i*n + j] *= scale;
        }
    }
    return 0;
}

int solve_jacobi_iterative(double *matrix, size_t n, size_t m, double *b, double *startvector) {
    double *x;
    x = calloc(m, sizeof(double));

    for (int k = 0; k < 20; ++k) {
        x = memcpy(x, b, m * sizeof(double));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    x[i] = x[i] - matrix[i*n + j] * startvector[j];
                }
            }
            x[i] = x[i] / matrix[i*n + i];
        }
        startvector = memcpy(startvector, x, m * sizeof(double));
        printf("Solution after step %i:\n", k);
        print_vector(startvector, n);
    }
    
    return 0;
}

int solve_jacobi_threaded(double *matrix, size_t n, size_t m, double *b, double *startvector) {
    double *x;
    x = calloc(m, sizeof(double));

    for (int k = 0; k < 20; ++k) {
        x = memcpy(x, b, m * sizeof(double));
        int i, j;
        #pragma omp parallel for shared(x) private(i,j)
        for (i = 0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
                if (j != i) {
                    x[i] = x[i] - matrix[i * n + j] * startvector[j];
                }
            }
            x[i] = x[i] / matrix[i * n + i];
        }
        startvector = memcpy(startvector, x, m * sizeof(double));
        printf("Solution after step %i:\n", k);
        print_vector(startvector, n);
    }

    return 0;
}

int jac() {
    size_t n, m;
    n = 2;
    double *A, *b, *x;
    A = calloc(n * n, sizeof(double));
    b = malloc(n * sizeof(double));
    x = malloc(n * sizeof(double));

    clock_t begin, end_i, end_t;

    A[0] = 2;
    A[1] = 1;
    A[2] = 5;
    A[3] = 7;

    double C[4][4] = {2,1,5,7};
    double d[4] = {11, 13};

    b[0] = 11;
    b[1] = 13;

    x[0] = 1;
    x[1] = 1;

    printf("Matrix A:\n");
    print_matrix(&C, n, n);
    printf("Vector b:\n");
    print_vector(d, n);
    begin = clock();
    solve_jacobi_iterative(&C, n, n, d, x);
    end_i = clock() - begin;
    x[0] = 1;
    x[0] = 1;
    begin = clock();
    solve_jacobi_threaded(&C, n, n, d, x);
    end_t = clock() - begin;
    printf("Solution:\n");
    print_vector(x, n);

    printf("delta iterative: %d\n", (int) end_i);
    printf("seconds iterative: %f\n", ((end_i * 1.) / CLOCKS_PER_SEC));
    printf("delta threaded: %d\n", (int) end_t);
    printf("seconds threaded: %f\n", ((end_t * 1.) / CLOCKS_PER_SEC));
    printf("relative speedup: %f\n", (double) end_i / end_t);
    printf("CLOCKS_PER_SEC = %lu\n", CLOCKS_PER_SEC);

    free(A); free(b); free(x);
    return 0;
}

int sharpen(unsigned char *data, unsigned char *output, int x, int y, int n, int p){
    memcpy(output, data, (size_t) x * y * n);
    int yj, xi, c;
    signed int sum;
    #pragma omp parallel if(p) shared(output) private(yj,xi,c,sum)
    #pragma omp for
    for (yj = 1; yj < y-1; ++yj) {
        for (xi = 1; xi < x-1; ++xi) {
            for (c = 0; c < n; ++c) {
                // Apply kernel
                sum = -1 * (signed int) data[(yj-1) * x * n + (xi-1) * n + c] + -1 * (signed int) data[(yj-1) * x * n + (xi) * n + c] + -1 * (signed int) data[(yj-1) * x * n + (xi+1) * n + c] +
                      -1 * (signed int) data[(yj) * x * n + (xi-1) * n + c] + 9 * (signed int) data[(yj) * x * n + (xi) * n + c] + -1 * (signed int) data[(yj) * x * n + (xi+1) * n + c] +
                      -1 * (signed int) data[(yj+1) * x * n + (xi-1) * n + c] + -1 * (signed int) data[(yj+1) * x * n + (xi) * n + c] + -1 * (signed int) data[(yj+1) * x * n + (xi+1) * n + c];

                // Set stuff
                output[(yj * x * n) + (xi * n) + c] = (unsigned char) CLAMP(sum, 0, 255);
            }
        }
    }
    return 0;
}

int image(char *path) {
    int x,y,n;
    unsigned char *data = stbi_load(path, &x, &y, &n, 0);
    if (data == NULL) {
        printf("%s\n", stbi_failure_reason());
        return -1;
    }
    printf("x:%i,y:%i,composite layers:%i\n", x,y,n);

    unsigned char *output = malloc((size_t) x * y * n);

    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_REALTIME, &start);
    sharpen(data, output, x, y, n, 0);
    clock_gettime(CLOCK_REALTIME, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("sharpen iterative:\t%f\n", elapsed);

    clock_gettime(CLOCK_REALTIME, &start);
    sharpen(data, output, x, y, n, 1);
    clock_gettime(CLOCK_REALTIME, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("sharpen threaded:\t%f\n", elapsed);

    if (stbi_write_png("out.png", x, y, n, output, 0) == 0){
        perror("Error saving file");
        return -1;
    }
    stbi_image_free(data);
    free(output);
    return 0;
}

int main(int argc, char *argv[]) {
    if(argc < 2) {
        printf("Specify path to image!");
        return -1;
    }
    printf("Max threads: %i\n", omp_get_max_threads());
    return image(argv[1]);
}