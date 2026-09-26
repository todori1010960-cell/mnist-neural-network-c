#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stddef.h>

void print(int m, int n, const float *x);
void fc(int m, int n, const float *x, const float *A, const float *b, float *y);
void relu(int n, const float *x, float *y);
void softmax(int n, const float *x, float *y);
int argmax(int n, const float *values);

void forward6(const float *A1, const float *b1,
              const float *A2, const float *b2,
              const float *A3, const float *b3,
              const float *x, float *y);
void forward6_train(const float *A1, const float *b1,
                    const float *A2, const float *b2,
                    const float *A3, const float *b3,
                    const float *x,
                    float *u1, float *z1, float *u2,
                    float *z2, float *u3, float *y);
float cross_entropy_error(const float *y, int t);
void backward6(const float *A1, const float *b1,
               const float *A2, const float *b2,
               const float *A3, const float *b3,
               const float *x, unsigned char t, float *y,
               float *dEdA1, float *dEdb1,
               float *dEdA2, float *dEdb2,
               float *dEdA3, float *dEdb3);

void init(int n, float x, float *o);
void add(int n, const float *x, float *o);
void scale(int n, float x, float *o);
void rand_init(int n, float *o);
void rand_init_fc(int m, int n, float *A);
void shuffle(int n, int *x);

void evaluate6(const char *name,
               const float *A1, const float *b1,
               const float *A2, const float *b2,
               const float *A3, const float *b3,
               const float *images, const unsigned char *labels,
               int count);
void train6(float *A1, float *b1,
            float *A2, float *b2,
            float *A3, float *b3,
            const float *train_x, const unsigned char *train_y,
            int train_count,
            const float *test_x, const unsigned char *test_y,
            int test_count);

void save(const char *filename, int m, int n,
          const float *A, const float *b);
void load(const char *filename, int m, int n,
          float *A, float *b);

#endif
