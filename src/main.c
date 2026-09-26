#include "mnist_loader.h"
#include "neural_network.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void allocate_parameters(float **A1, float **b1,
                                float **A2, float **b2,
                                float **A3, float **b3)
{
    *A1 = malloc(sizeof **A1 * 50 * 784);
    *b1 = malloc(sizeof **b1 * 50);
    *A2 = malloc(sizeof **A2 * 100 * 50);
    *b2 = malloc(sizeof **b2 * 100);
    *A3 = malloc(sizeof **A3 * 10 * 100);
    *b3 = malloc(sizeof **b3 * 10);
    if (*A1 == NULL || *b1 == NULL || *A2 == NULL ||
        *b2 == NULL || *A3 == NULL || *b3 == NULL) {
        fprintf(stderr, "malloc failed while allocating parameters\n");
        exit(1);
    }
}

static void free_parameters(float *A1, float *b1,
                            float *A2, float *b2,
                            float *A3, float *b3)
{
    free(A1);
    free(b1);
    free(A2);
    free(b2);
    free(A3);
    free(b3);
}

static void run_train(int argc, char **argv)
{
    MnistDataset dataset;
    float *A1, *b1, *A2, *b2, *A3, *b3;

    if (argc != 6) {
        fprintf(stderr,
                "usage: %s train DATA_DIR fc1.dat fc2.dat fc3.dat\n",
                argv[0]);
        exit(1);
    }
    if (!mnist_load(argv[2], &dataset)) {
        exit(1);
    }
    allocate_parameters(&A1, &b1, &A2, &b2, &A3, &b3);
    srand(0);
    rand_init_fc(50, 784, A1);
    init(50, 0.0f, b1);
    rand_init_fc(100, 50, A2);
    init(100, 0.0f, b2);
    rand_init_fc(10, 100, A3);
    init(10, 0.0f, b3);

    printf("before training\n");
    evaluate6("train", A1, b1, A2, b2, A3, b3,
               dataset.train_x, dataset.train_y, dataset.train_count);
    evaluate6("test ", A1, b1, A2, b2, A3, b3,
               dataset.test_x, dataset.test_y, dataset.test_count);
    train6(A1, b1, A2, b2, A3, b3,
           dataset.train_x, dataset.train_y, dataset.train_count,
           dataset.test_x, dataset.test_y, dataset.test_count);
    save(argv[3], 50, 784, A1, b1);
    save(argv[4], 100, 50, A2, b2);
    save(argv[5], 10, 100, A3, b3);
    free_parameters(A1, b1, A2, b2, A3, b3);
    mnist_free(&dataset);
}

static void run_infer(int argc, char **argv)
{
    MnistDataset dataset;
    float *A1, *b1, *A2, *b2, *A3, *b3;
    char *end;
    long index;
    float y[10];

    if (argc != 7) {
        fprintf(stderr,
                "usage: %s infer DATA_DIR fc1.dat fc2.dat fc3.dat INDEX\n",
                argv[0]);
        exit(1);
    }
    errno = 0;
    index = strtol(argv[6], &end, 10);
    if (errno != 0 || *end != '\0' || index < 0) {
        fprintf(stderr, "invalid test image index: %s\n", argv[6]);
        exit(1);
    }
    if (!mnist_load(argv[2], &dataset)) {
        exit(1);
    }
    if (index >= dataset.test_count) {
        fprintf(stderr, "test image index is out of range: %ld\n", index);
        mnist_free(&dataset);
        exit(1);
    }
    allocate_parameters(&A1, &b1, &A2, &b2, &A3, &b3);
    load(argv[3], 50, 784, A1, b1);
    load(argv[4], 100, 50, A2, b2);
    load(argv[5], 10, 100, A3, b3);
    forward6(A1, b1, A2, b2, A3, b3,
             dataset.test_x + 784 * index, y);
    printf("test index = %ld, label = %d\n", index,
           dataset.test_y[index]);
    printf("probabilities:\n");
    print(1, 10, y);
    printf("answer = %d\n", argmax(10, y));
    free_parameters(A1, b1, A2, b2, A3, b3);
    mnist_free(&dataset);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "usage:\n"
                "  %s train DATA_DIR fc1.dat fc2.dat fc3.dat\n"
                "  %s infer DATA_DIR fc1.dat fc2.dat fc3.dat INDEX\n",
                argv[0], argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "train") == 0) {
        run_train(argc, argv);
    } else if (strcmp(argv[1], "infer") == 0) {
        run_infer(argc, argv);
    } else {
        fprintf(stderr, "unknown mode: %s\n", argv[1]);
        return 1;
    }
    return 0;
}
