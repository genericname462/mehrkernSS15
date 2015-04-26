#include "async.h"

int logger(int *run, char *shared_buffer) {
    while ((*run) < 5) {
        printf("Logger: logging... run: %i\n", (*run));
        sleep(2);
        #pragma omp atomic update
        ++(*run);
    }


    printf("Logger: stopping\n");
    return 0;
}

int user_input(int *run, char *shared_buffer) {
    while ((*run) < 5) {
        fgets(shared_buffer, 19, stdin);
        //printf("user_input: run: %i\n", (*run));
    }


    printf("User input: stopping\n");
    return 0;
}

int printer(int *run, char *shared_buffer) {
    while ((*run) < 5) {
        printf("Printer: run: %i %s\n", (*run), shared_buffer);
        sleep(2);
    }


    printf("Printer: stopping\n");
    return 0;
}

int async_demo(int num_threads) {

    int run = 1;
    char *shared_buffer = calloc(20 , 1);

    #pragma omp parallel shared(run)
    {
        #pragma omp single nowait
        {
            #pragma omp task shared(run, shared_buffer) untied
            {
                //user_input(&run, shared_buffer);
            }
            #pragma omp task shared(run, shared_buffer) untied
            {
                logger(&run, shared_buffer);
            }
            #pragma omp task shared(run, shared_buffer) untied
            {
                printer(&run, shared_buffer);
            }
        }
    }
    #pragma omp taskwait
    printf("run final: %i", run);
    return 0;
}

