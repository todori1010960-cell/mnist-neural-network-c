#include "mnist_loader.h"

#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_u32(FILE *fp, uint32_t *value)
{
    unsigned char bytes[4];
    if (fread(bytes, 1, sizeof bytes, fp) != sizeof bytes) {
        return 0;
    }
    *value = ((uint32_t)bytes[0] << 24) |
             ((uint32_t)bytes[1] << 16) |
             ((uint32_t)bytes[2] << 8) |
             (uint32_t)bytes[3];
    return 1;
}

static int make_path(char *path, size_t size,
                     const char *directory, const char *name)
{
    size_t length = strlen(directory);
    int needs_slash = length > 0 && directory[length - 1] != '/' &&
                      directory[length - 1] != '\\';
    int written = snprintf(path, size, "%s%s%s", directory,
                           needs_slash ? "/" : "", name);
    return written >= 0 && (size_t)written < size;
}

static int read_images(const char *path, float **images,
                       int *count, int *width, int *height)
{
    FILE *fp = fopen(path, "rb");
    uint32_t magic, number, rows, columns;
    unsigned char *pixels = NULL;
    float *result = NULL;
    size_t pixel_count;

    if (fp == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return 0;
    }
    if (!read_u32(fp, &magic) || !read_u32(fp, &number) ||
        !read_u32(fp, &rows) || !read_u32(fp, &columns) ||
        magic != 2051 || rows == 0 || columns == 0 ||
        number > (uint32_t)INT_MAX) {
        fprintf(stderr, "invalid MNIST image file: %s\n", path);
        fclose(fp);
        return 0;
    }

    pixel_count = (size_t)number * rows * columns;
    pixels = malloc(pixel_count);
    result = malloc(pixel_count * sizeof *result);
    if (pixels == NULL || result == NULL ||
        fread(pixels, 1, pixel_count, fp) != pixel_count) {
        fprintf(stderr, "failed to read MNIST images: %s\n", path);
        free(pixels);
        free(result);
        fclose(fp);
        return 0;
    }
    fclose(fp);

    for (size_t i = 0; i < pixel_count; i++) {
        result[i] = (float)pixels[i] / 255.0f;
    }
    free(pixels);
    *images = result;
    *count = (int)number;
    *width = (int)columns;
    *height = (int)rows;
    return 1;
}

static int read_labels(const char *path, unsigned char **labels, int *count)
{
    FILE *fp = fopen(path, "rb");
    uint32_t magic, number;
    unsigned char *result;

    if (fp == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return 0;
    }
    if (!read_u32(fp, &magic) || !read_u32(fp, &number) ||
        magic != 2049 || number > (uint32_t)INT_MAX) {
        fprintf(stderr, "invalid MNIST label file: %s\n", path);
        fclose(fp);
        return 0;
    }
    result = malloc(number);
    if (result == NULL || fread(result, 1, number, fp) != number) {
        fprintf(stderr, "failed to read MNIST labels: %s\n", path);
        free(result);
        fclose(fp);
        return 0;
    }
    fclose(fp);
    *labels = result;
    *count = (int)number;
    return 1;
}

static int load_split(const char *directory, const char *image_name,
                      const char *label_name, float **images,
                      unsigned char **labels, int *count)
{
    char image_path[1024];
    char label_path[1024];
    int image_count, label_count, width, height;

    if (!make_path(image_path, sizeof image_path, directory, image_name) ||
        !make_path(label_path, sizeof label_path, directory, label_name) ||
        !read_images(image_path, images, &image_count, &width, &height) ||
        !read_labels(label_path, labels, &label_count)) {
        return 0;
    }
    if (width != 28 || height != 28 || image_count != label_count) {
        fprintf(stderr, "MNIST image/label dimensions do not match\n");
        free(*images);
        free(*labels);
        *images = NULL;
        *labels = NULL;
        return 0;
    }
    *count = image_count;
    return 1;
}

int mnist_load(const char *directory, MnistDataset *dataset)
{
    memset(dataset, 0, sizeof *dataset);
    if (!load_split(directory, "train-images-idx3-ubyte",
                    "train-labels-idx1-ubyte", &dataset->train_x,
                    &dataset->train_y, &dataset->train_count) ||
        !load_split(directory, "t10k-images-idx3-ubyte",
                    "t10k-labels-idx1-ubyte", &dataset->test_x,
                    &dataset->test_y, &dataset->test_count)) {
        mnist_free(dataset);
        return 0;
    }
    return 1;
}

void mnist_free(MnistDataset *dataset)
{
    free(dataset->train_x);
    free(dataset->train_y);
    free(dataset->test_x);
    free(dataset->test_y);
    memset(dataset, 0, sizeof *dataset);
}
