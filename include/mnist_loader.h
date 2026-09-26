#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

typedef struct {
    float *train_x;
    unsigned char *train_y;
    int train_count;
    float *test_x;
    unsigned char *test_y;
    int test_count;
} MnistDataset;

int mnist_load(const char *directory, MnistDataset *dataset);
void mnist_free(MnistDataset *dataset);

#endif
