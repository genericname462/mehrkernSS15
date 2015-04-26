#ifndef OPENMP_ASYNC_H
#define OPENMP_ASYNC_H


#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <sys/unistd.h>


int logger(int *run, char *shared_buffer);

int user_input(int *run, char *shared_buffer);

int printer(int *run, char *shared_buffer);

int async_demo(int num_threads);








#endif //OPENMP_ASYNC_H
