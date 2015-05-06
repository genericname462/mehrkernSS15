#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <omp.h>
#include <string.h>
#define _POSIX_C_SOURCE 199309L
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "../lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

//#include "async.h"
#include "imagefilter.h"

#define push(sp, n) (*((sp)++) = (n))
#define pop(sp) (*--(sp))

uint32_t color_map[] = {0xffff0000,0xffff00ff,0xff00ff00,0xff00ffff,0xffb0acff,0xffff007B, 0xffffff00, 0xff4b88ff};

struct position {
    int x;
    int y;
};
typedef struct position position;

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

int jacobi_demo() {
    size_t n;
    n = 4;
    double *A, *b, *x;
    A = calloc(n * n, sizeof(double));
    b = malloc(n * sizeof(double));
    x = malloc(n * sizeof(double));

    clock_t begin, end_i, end_t;

    A[0] = 2;
    A[1] = 1;
    A[2] = 5;
    A[3] = 7;

    double C[2][2] = {{2,1},{5,7}};
    double d[2] = {11, 13};

    b[0] = 11;
    b[1] = 13;

    x[0] = 1;
    x[1] = 1;

    printf("Matrix A:\n");
    print_matrix((double*)&C, n, n);
    printf("Vector b:\n");
    print_vector(d, n);
    begin = clock();
    solve_jacobi_iterative((double*)&C, n, n, d, x);
    end_i = clock() - begin;
    x[0] = 1;
    x[0] = 1;
    begin = clock();
    solve_jacobi_threaded((double*)&C, n, n, d, x);
    end_t = clock() - begin;
    printf("Solution:\n");
    print_vector(x, n);

    printf("delta iterative: %d\n", (int) end_i);
    printf("seconds iterative: %f\n", ((end_i * 1.) / CLOCKS_PER_SEC));
    printf("delta threaded: %d\n", (int) end_t);
    printf("seconds threaded: %f\n", ((end_t * 1.) / CLOCKS_PER_SEC));
    printf("relative speedup: %f\n", (double) end_i / end_t);
    printf("CLOCKS_PER_SEC = %i\n", CLOCKS_PER_SEC);

    free(A); free(b); free(x);
    return 0;
}

int image_demo(char *path, int save) {
    int x,y,n;
    unsigned char *data = stbi_load(path, &x, &y, &n, 4);
    if (data == NULL) {
        printf("%s\n", stbi_failure_reason());
        return -1;
    }
    printf("Imagedata: x:%i,y:%i,composite layers:%i\n", x,y,n);

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


    apply_kernel_to_image(data, output, x, y, n, 1, (double*) &sharpen_kernel, 3);
    if (save && stbi_write_png("image_sharpend.png", x, y, n, output, 0) == 0){
        perror("Error saving file");
        return -1;
    }
    apply_kernel_to_image(data, output, x, y, n, 1, (double*) &gauss_kernel, 7);
    if (save && stbi_write_png("image_gauss.png", x, y, n, output, 0) == 0){
        perror("Error saving file");
        return -1;
    }

    stbi_image_free(data);
    free(output);
    return 0;
}

int solve_maze_partition(unsigned char *data, unsigned char *output, int x, int y, int n, int parallel) {
    uint32_t wall = 0xff000000;    //black
    uint32_t path = 0xffffffff;    //white

    memcpy(output, data, (size_t) x * y * n);

    //Pixel format on little endian: [ALPHA | BLUE | GREEN | RED ]
    uint32_t (*out_m)[y][x];
    out_m = (void*) output;

    //Find entry and exit
    //TODO: unlimit image size. MAX_INT at the moment
    position ent = {.x = -1, .y = -1};
    position ex = {.x = -1, .y = -1};

    for (int j = 0; j < y; ++j) {
        for (int i = 0; i < x; ++i) {
            if (i == 0 || j == 0 || i == x-1 || j == y-1) {
                if ((*out_m)[j][i] == path) {
                    if (ent.x == -1) {
                        ent.x = i;
                        ent.y = j;
                    } else {
                        ex.x = i;
                        ex.y = j;
                        break;
                    }
                }
            }
        }
    }
    printf("entry position: (%i,%i)\texit position: (%i,%i)\n", ent.x, ent.y, ex.x, ex.y);
    printf("entry color: %x\texit color: %x\n", (*out_m)[ent.y][ent.x], (*out_m)[ex.y][ex.x]);
    printf("Using partition!\n");
    int done = 0;

    //testing purpose, simulate the other thread
    //(*out_m)[ex.y][ex.x] = 0xffff00ff;
    //(*out_m)[ent.y][ent.x] = 0xff00ffff;

    //More than 2 threads makes detection of success costly and is not implemented
    #pragma omp parallel num_threads(2) shared(out_m, done)
    {
        //Init stuff
        uint32_t self_color = color_map[omp_get_thread_num()];
        position current_position;
        position stack[100000]; //Fix later
        position *sbp = stack;
        position *sp = stack;

        //Set start position
        if (omp_get_thread_num() == 0) {
            push(sp, ent);
        } else if (omp_get_thread_num() == 1) {
            push(sp, ex);
        }
        while (sp > sbp && !done) {
            //Get new cell, test if visited. Terminate if marked by another thread, mark self if not
            current_position = pop(sp);
            if ((*out_m)[current_position.y][current_position.x] != path && (*out_m)[current_position.y][current_position.x] != self_color) {
                printf("Thread %i: found other thread at: (%i,%i), color: %x\n", omp_get_thread_num(), current_position.x, current_position.y, (*out_m)[current_position.y][current_position.x]);
            #pragma omp atomic write
                done = 1;
            } else {
                (*out_m)[current_position.y][current_position.x] = self_color;

                //Find new cells
                position north = {current_position.x, current_position.y - 1};
                position east = {current_position.x + 1, current_position.y};
                position south = {current_position.x, current_position.y + 1};
                position west = {current_position.x - 1, current_position.y};
                if (north.y >= 0 && north.y < y &&
                    (*out_m)[north.y][north.x] != wall && (*out_m)[north.y][north.x] != self_color) {
                    push(sp, north);
                }
                if (east.x >= 0 && east.x < x &&
                    (*out_m)[east.y][east.x] != wall && (*out_m)[east.y][east.x] != self_color) {
                    push(sp, east);
                }
                if (south.y >= 0 && south.y < y &&
                    (*out_m)[south.y][south.x] != wall && (*out_m)[south.y][south.x] != self_color) {
                    push(sp, south);
                }
                if (west.x >= 0 && west.x < x &&
                    (*out_m)[west.y][west.x] != wall && (*out_m)[west.y][west.x] != self_color) {
                    push(sp, west);
                }
            }
        }
        //TODO: Print final path

        printf("Thread %i: done, stack size: %lu\n", omp_get_thread_num(), sp - sbp);
    }
    return 0;
}

int solve_maze_dead_end_elimination(unsigned char *data, unsigned char *output, int x, int y, int n, int parallel) {
    uint32_t wall = 0xff000000;    //black
    uint32_t path = 0xffffffff;    //white

    memcpy(output, data, (size_t) x * y * n);

    //Pixel format on little endian: [ ALPHA | BLUE | GREEN | RED ]
    uint32_t (*out_m)[y][x];
    out_m = (void*) output;

    //Find entry and exit
    //TODO: unlimit image size. MAX_INT at the moment
    position ent = {.x = -1, .y = -1};
    position ex = {.x = -1, .y = -1};

    for (int j = 0; j < y; ++j) {
        for (int i = 0; i < x; ++i) {
            if (i == 0 || j == 0 || i == x-1 || j == y-1) {
                if ((*out_m)[j][i] == path) {
                    if (ent.x == -1) {
                        ent.x = i;
                        ent.y = j;
                    } else {
                        ex.x = i;
                        ex.y = j;
                        break;
                    }
                }
            }
        }
    }
    printf("entry position: (%i,%i)\texit position: (%i,%i)\n", ent.x, ent.y, ex.x, ex.y);
    printf("entry color: %x\texit color: %x\n", (*out_m)[ent.y][ent.x], (*out_m)[ex.y][ex.x]);

    if (!parallel) {
        //Algorithm
        printf("Using sequential dead end elimination!\n");
        // ## SEQUENTIAL VERSION ##
        int done = 1;
        int count = 0;
        while (done) {
            done = 0;
            for (int j = 1; j < y - 1; ++j) {
                for (int i = 1; i < x - 1; ++i) {
                    ++count;
                    if ((*out_m)[j][i] == path) {
                        int ends = 0;
                        if ((*out_m)[j][i + 1] == wall) {
                            ++ends;
                        }
                        if ((*out_m)[j + 1][i] == wall) {
                            ++ends;
                        }
                        if ((*out_m)[j][i - 1] == wall) {
                            ++ends;
                        }
                        if ((*out_m)[j - 1][i] == wall) {
                            ++ends;
                        }
                        if (ends >= 3) {
                            //printf("Dead end found!\n");
                            (*out_m)[j][i] = wall;
                            done = 1;
                        }
                    }
                }
            }
        }
        printf("count: %i\n", count);
    } else {
        printf("Using parallel dead end elimination!\n");

        #pragma omp parallel
        {
            //Init stuff
            position *stack = malloc(1000000 * sizeof(position));
            position *sbp = stack;
            position *sp = stack;
            uint32_t self_color = color_map[omp_get_thread_num()];

            //Set work range per thread
            int num_th = omp_get_num_threads();
            int min_x, max_x, min_y, max_y;
            min_x = CLAMP((x / num_th) * omp_get_thread_num(), 1, x - 1);
            max_x = CLAMP(min_x + x / num_th, 1, x - 1);
            min_y = 1;
            max_y = y - 1;

            //Populate the stack so other threads can't cut this one of by surrounding it
            printf("Thread %i: x-range %i - %i\ty-range %i - %i\n", omp_get_thread_num(), min_x, max_x, min_y, max_y);
            position end;
            for (int j = min_y; j < max_y; ++j) {
                for (int i = min_x; i < max_x; ++i) {
                    if ((*out_m)[j][i] == path) {
                        int ends = 0;
                        if ((*out_m)[j][i + 1] != path) {
                            ++ends;
                        }
                        if ((*out_m)[j + 1][i] != path) {
                            ++ends;
                        }
                        if ((*out_m)[j][i - 1] != path) {
                            ++ends;
                        }
                        if ((*out_m)[j - 1][i] != path) {
                            ++ends;
                        }
                        if (ends >= 3) {
                            //printf("Dead end found!\n");
                            (*out_m)[j][i] = self_color;
                            end.x = i; end.y = j;
                            push(sp, end);
                        }
                    }
                }
            }

            int count = 0;
            position current_position;
            int north_c, east_c, south_c, west_c;
            while (sp > sbp) {
                ++count;
                //printf("%lu dead ends on stack\n", sp - sbp);
                //stbi_write_png("solution_maze_temp.png", x, y, n, output, 0);
                current_position = pop(sp);
                //printf("current: (%i,%i)\n", current_position.x, current_position.y);
                //(*out_m)[current.y][current.x] = 0xff00ff00;
                north_c=0;east_c=0;south_c=0;west_c=0;

                //Find new cells
                position north = {current_position.x, current_position.y - 1};
                position east = {current_position.x + 1, current_position.y};
                position south = {current_position.x, current_position.y + 1};
                position west = {current_position.x - 1, current_position.y};
                if (north.y >= 0 && north.y < y &&
                    (*out_m)[north.y][north.x] != path) {
                    north_c = 1;
                }
                if (east.x >= 0 && east.x < x &&
                    (*out_m)[east.y][east.x] != path) {
                    east_c = 1;
                }
                if (south.y >= 0 && south.y < y &&
                    (*out_m)[south.y][south.x] != path) {
                    south_c = 1;
                }
                if (west.x >= 0 && west.x < x &&
                    (*out_m)[west.y][west.x] != path) {
                    west_c = 1;
                }

                if (west_c + east_c + south_c + north_c >= 3) {
                    (*out_m)[current_position.y][current_position.x] = self_color;
                    if (!east_c)
                        push(sp, east);
                    if (!south_c)
                        push(sp, south);
                    if (!west_c)
                        push(sp, west);
                    if (!north_c)
                        push(sp, north);
                }
            }
            free(stack);
        }
    }
    return 0;
}

int maze_demo(char *path, int save) {
    int x,y,n;
    unsigned char *data = stbi_load(path, &x, &y, &n, 4);
    if (data == NULL) {
        printf("%s\n", stbi_failure_reason());
        return -1;
    }
    printf("Imagedata: x:%i, y:%i, composite layers:%i\n", x,y,n);

    unsigned char *output = calloc((size_t) x * y * n, 1);

    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_REALTIME, &start);
    //solve_maze_partition(data, output, x, y, n, 0);
    solve_maze_dead_end_elimination(data, output, x, y, n, 1);
    clock_gettime(CLOCK_REALTIME, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Maze solved in %f seconds\n", elapsed);

//    clock_gettime(CLOCK_REALTIME, &start);
//    solve_maze_partition(data, output, x, y, n, 1);
//    clock_gettime(CLOCK_REALTIME, &finish);
//    elapsed = (finish.tv_sec - start.tv_sec);
//    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
//    printf("maze threaded:\t%f\n", elapsed);

    if (save && stbi_write_png("solution_maze.png", x, y, n, output, 0) == 0){
        perror("Error saving file");
        return -1;
    }
    stbi_image_free(data);
    free(output);
    return 0;
}

int main(int argc, char *argv[]) {
    if(argc < 2) {
        printf("Specify path to image or demo name!\n");
        return -1;
    }
    if (!strncmp(argv[1], "async_demo", 20)){
        //return async_demo(1);
    }
    printf("Max threads: %i\n", omp_get_max_threads());
    printf("Max processors: %i\n", omp_get_num_procs());

    return image_demo(argv[1], 1);
    //return maze_demo(argv[1], 0);

}