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


#define CLAMP(x, low, high) ({\
  __typeof__(x) __x = (x); \
  __typeof__(low) __low = (low);\
  __typeof__(high) __high = (high);\
  __x > __high ? __high : (__x < __low ? __low : __x);\
  })

#define push(sp, n) (*((sp)++) = (n))
#define pop(sp) (*--(sp))

uint32_t color_map[] = {0xffff0000,0xffff00ff,0xff00ff00,0xff00ffff};

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

int sharpen(unsigned char *data, unsigned char *output, int x, int y, int n, int parallel){
    memcpy(output, data, (size_t) x * y * n);
    int yj, xi, c;
    signed int sum;
    #pragma omp parallel if(parallel) shared(output) private(yj,xi,c,sum)
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

int image(char *path, int save) {
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

    if (save && stbi_write_png("image_sharpen.png", x, y, n, output, 0) == 0){
        perror("Error saving file");
        return -1;
    }
    stbi_image_free(data);
    free(output);
    return 0;
}

int solve_maze(unsigned char *data, unsigned char *output, int x, int y, int n, int parallel) {
    uint32_t wall = 0xff000000;    //black
    uint32_t path = 0xffffffff;    //white

    //Pixel format on little endian: [ALPHA | BLUE | GREEN | RED ]
    uint32_t *bigdata;
    bigdata = (uint32_t*) data;

    uint32_t *bigout;
    bigout = (uint32_t*) output;

    memcpy(output, data, (size_t) x * y * n);

    //Find entry and exit
    //TODO: unlimit image size. MAX_INT at the moment
    int entry = -1;
    int exit = -1;
    for (int j = 0; j < y; ++j) {
        for (int i = 0; i < x; ++i) {
            if (i == 0 || j == 0 || i == x-1 || j == y-1) {
                if (bigout[j * x + i] == path) {
                    if (entry == -1) {
                        entry = j * x + i;
                        bigout[j * x + i] = 0xff0000ff; //red
                    } else {
                        exit = j * x + i;
                        bigout[j * x + i] = 0xffffff00; //cyan
                        break;
                    }
                }
            }
        }
    }
    printf("entry: %i\texit: %i\n", entry, exit);

    //Algorithm
    int max_size = x * y;
    int stack[100000]; //Fix later, also mark access as critical
    int *sbp = stack;
    int *sp = stack;
    int v;

    push(sp, entry);
    omp_set_num_threads(1);
    //#pragma omp parallel if(parallel) default(shared) private(v)
    {
        while (sp >= sbp) {
            //test if not visited already
            //#pragma omp critical (access_stack)
            {
                v = pop(sp);
            }
            //printf("current pos: %i\n", v);
            if (bigout[v] == path || bigout[v] == 0xff0000ff) {
                //printf("%i not visited!\n", v);
                //bigout[v] = visited;
                //#pragma omp critical (access_image)
                {
                    bigout[v] = color_map[omp_get_thread_num()];
                }
                //push all adjacent paths to the stack
                if (v + 1 >= 0 && v + 1 < max_size && bigout[v + 1] != wall) { //EAST
                    if (bigout[v + 1] == 0xffffff00) {
                        printf("Found exit: %i\n", v + 1);
                        break;
                    }
                    //printf("East free!\n");
                    //#pragma omp critical (access_stack)
                    {
                        push(sp, v + 1);
                    }
                }
                if (v + x >= 0 && v + x <= max_size && bigout[v + x] != wall) { //SOUTH
                    if (bigout[v + x] == 0xffffff00) {
                        printf("Found exit: %i\n", v + x);
                        break;
                    }
                    //printf("South free!\n");
                    //#pragma omp critical (access_stack)
                    {
                        push(sp, v + x);
                    }
                }
                if (v - 1 >= 0 && v - 1 < max_size && bigout[v - 1] != wall) { //WEST
                    if (bigout[v - 1] == 0xffffff00) {
                        printf("Found exit: %i\n", v - 1);
                        break;
                    }
                    //printf("West free!\n");
                    //#pragma omp critical (access_stack)
                    {
                        push(sp, v - 1);
                    }
                }
                if (v - x >= 0 && v - x < max_size && bigout[v - x] != wall) { //NORTH
                    if (bigout[v - x] == 0xffffff00) {
                        printf("Found exit: %i\n", v - x);
                        break;
                    }
                    //printf("North free!\n");
                    //#pragma omp critical (access_stack)
                    {
                        push(sp, v - x);
                    }
                }
            }
        }
    }

    return 0;
}

int solve_maze_dead_end_elimination(unsigned char *data, unsigned char *output, int x, int y, int n, int parallel) {
    uint32_t wall = 0xff000000;    //black
    uint32_t path = 0xffffffff;    //white

    //Pixel format on little endian: [ALPHA | BLUE | GREEN | RED ]

    uint32_t *bigout;
    bigout = (uint32_t*) output;

    memcpy(output, data, (size_t) x * y * n);

    uint32_t (*out_m)[y][x];
    out_m = (void*) output;

    struct position {
        int x;
        int y;
    };
    typedef struct position position;

    //Find entry and exit
    //TODO: unlimit image size. MAX_INT at the moment
    int entry = -1;
    int exit = -1;
    position ent = {.x = -1, .y = -1};
    position ex = {.x = -1, .y = -1};

    for (int j = 0; j < y; ++j) {
        for (int i = 0; i < x; ++i) {
            if (i == 0 || j == 0 || i == x-1 || j == y-1) {
                if (bigout[j * x + i] == path) {
                    if (entry == -1) {
                        entry = j * x + i;
                        ent.x = i;
                        ent.y = j;
                        //(*out_m)[ent.y][ent.x] = 0xff0000ff;
                        //bigout[j * x + i] = 0xff0000ff; //red
                    } else {
                        exit = j * x + i;
                        ex.x = i;
                        ex.y = j;
                        //(*out_m)[ex.y][ex.x] = 0xffffff00;
                        //bigout[j * x + i] = 0xffffff00; //cyan
                        break;
                    }
                }
            }
        }
    }
    printf("entry position: (%i,%i)\texit position: (%i,%i)\n", ent.x, ent.y, ex.x, ex.y);
    printf("entry color: %x\texit color: %x\n", (*out_m)[ent.y][ent.x], (*out_m)[ex.y][ex.x]);

    //Algorithm
    if (!parallel) {
        printf("Using seq version!\n");
        // ## SEQUENTIAL VERSION ##
        int max_size = x * y;
        int done = 1;
        int count = 0;
        while (done) {
            done = 0;
            ++count;
            for (int j = 1; j < y - 1; ++j) {
                for (int i = 1; i < x - 1; ++i) {
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
    } else {
        printf("Using threaded version!\n");
        // ## THREADED VERSION ##
        int done = 0;

        //testing purpose, simulate other thread
        //(*out_m)[ex.y][ex.x] = 0xffff00ff;
        //(*out_m)[ent.y][ent.x] = 0xff00ffff;
        //(*out_m)[56][51] = 0xffff00ff;

        #pragma omp parallel num_threads(2) shared(done, out_m)
        {
            //Init stuff
            int cx, cy;
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
            while (!done && sp > sbp) {
                //Get new cell, test if visited. Terminate if marked by another thread, mark self if not
                current_position = pop(sp);

                if ((*out_m)[current_position.y][current_position.x] != path && (*out_m)[current_position.y][current_position.x] != self_color) {
                    printf("Other thread at: (%i,%i), color: %x\n", current_position.x, current_position.y, (*out_m)[current_position.y][current_position.x]);
                    #pragma omp atomic write
                        done = 1;
                    printf("done, %i, stack size: %lu\n", omp_get_thread_num(), sp - sbp);
                } else {
                    (*out_m)[current_position.y][current_position.x] = self_color;
                    //Find new cells
                    position north = {current_position.x, current_position.y - 1};
                    position east = {current_position.x + 1, current_position.y};
                    position south = {current_position.x, current_position.y + 1};
                    position west = {current_position.x - 1, current_position.y};

                    if (north.x >= 0 && north.y >= 0 && north.x < x && north.y < y &&
                            (*out_m)[north.y][north.x] != wall && (*out_m)[north.y][north.x] != self_color) {
                        //printf("north: %x\n", (*out_m)[north.y][north.x]);
                        push(sp, north);
                    }
                    if (east.x >= 0 && east.y >= 0 && east.x < x && east.y < y &&
                            (*out_m)[east.y][east.x] != wall && (*out_m)[east.y][east.x] != self_color) {
                        //printf("east: %x\n", (*out_m)[east.y][east.x]);
                        push(sp, east);
                    }
                    if (south.x >= 0 && south.y >= 0 && south.x < x && south.y < y &&
                            (*out_m)[south.y][south.x] != wall && (*out_m)[south.y][south.x] != self_color) {
                        //printf("south: %x\n", (*out_m)[south.y][south.x]);
                        push(sp, south);
                    }
                    if (west.x >= 0 && west.y >= 0 && west.x < x && west.y < y &&
                            (*out_m)[west.y][west.x] != wall && (*out_m)[west.y][west.x] != self_color) {
                        //printf("west: %x\n", (*out_m)[west.y][west.x]);
                        push(sp, west);
                    }
                }
            }
        }
        //Print final path

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
    //solve_maze(data, output, x, y, n, 0);
    solve_maze_dead_end_elimination(data, output, x, y, n, 1);
    clock_gettime(CLOCK_REALTIME, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("maze iterative:\t%f\n", elapsed);

//    clock_gettime(CLOCK_REALTIME, &start);
//    solve_maze(data, output, x, y, n, 1);
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
        printf("Specify path to image!");
        return -1;
    }
    printf("Max threads: %i\n", omp_get_max_threads());
    printf("Max processors: %i\n", omp_get_num_procs());

    //return image(argv[1], 0);
    return maze_demo(argv[1], 1);
}