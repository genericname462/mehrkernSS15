#include "async.h"

int logger(int *run, char *shared_buffer) {
    while ((*run)) {
        printf("Logger: logging... run: %i\n", (*run));
        sleep(2);
        #pragma omp atomic update
        --(*run);
    }


    printf("Logger: stopping\n");
    return 0;
}

int user_input(int *run, char *shared_buffer, omp_lock_t lock) {
    struct pollfd stdin_poll;
    stdin_poll.fd = STDIN_FILENO;
    stdin_poll.events = POLLIN | POLLRDBAND | POLLRDNORM | POLLPRI;
    while ((*run)) {
        if (poll(&stdin_poll, 1, 1000) == 1) {
            printf("Got data!\n");
            omp_set_lock(&lock);
            fgets(shared_buffer, 19, stdin);
            omp_unset_lock(&lock);
        }
        printf("user_input: run: %i\n", (*run));
    }


    printf("User input: stopping\n");
    return 0;
}

int printer(int *run, char *shared_buffer, omp_lock_t lock) {
    while ((*run)) {
        omp_set_lock(&lock);
        printf("Printer: run: %i %s\n", (*run), shared_buffer);
        omp_unset_lock(&lock);
        sleep(2);
    }


    printf("Printer: stopping\n");
    return 0;
}

int async_demo(int num_threads) {

    int run = 5;
    char *shared_buffer = calloc(20 , 1);
    omp_lock_t bufferlock;
    omp_init_lock(&bufferlock);

    #pragma omp parallel shared(run)
    {
        #pragma omp single nowait
        {
            #pragma omp task shared(run, shared_buffer) untied
            {

                user_input(&run, shared_buffer, bufferlock);
            }
            #pragma omp task shared(run, shared_buffer) untied
            {
                logger(&run, shared_buffer);
            }
            #pragma omp task shared(run, shared_buffer) untied
            {
                printer(&run, shared_buffer, bufferlock);
            }
        }
    }
    #pragma omp taskwait
    printf("run final: %i", run);
    omp_destroy_lock(&bufferlock);
    free(shared_buffer);
    return 0;
}

