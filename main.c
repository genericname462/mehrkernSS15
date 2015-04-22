#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <time.h>


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
        int i;
        #pragma omp parallel for shared(x) private(i)
        for (i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
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

int main() {
    size_t n, m;
    n = 4;
    //m = 2;
    double *A, *b, *x;
    A = calloc(n * n, sizeof(double));
    b = malloc(n * sizeof(double));
    x = malloc(n * sizeof(double));

    clock_t begin, end_i, end_t;

    A[0] = 2;
    A[1] = 1;
    A[2] = 5;
    A[3] = 7;

    double C[4][4] = {{1,2,4,7}, {2,4,8,2}, {9,4,8,3}, {1,3,8,7}};
    double d[4] = {12, 35, 2337, 1632};

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
};