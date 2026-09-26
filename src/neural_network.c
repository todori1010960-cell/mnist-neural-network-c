#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "neural_network.h"


//1次元配列xをm行n列として表示する デバッグ用部品
void print(int m,int n,const float *x){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%.4f",x[i*n+j]);
            if(j<n-1){
                putchar(' ');
            }
        }
        putchar('\n');
    }
}
//y=Ax+bを計算
void fc(int m,int n,const float *x,const float *A,const float *b,float *y){
    for(int i=0;i<m;i++){
        float sum=b[i];
        for(int j=0;j<n;j++){
            sum+=A[i*n+j]*x[j];
        }
        y[i]=sum;
    }
}

/*
ReLUを要素ごとに計算する。
*/
void relu(int n,const float *x,float *y){
    for(int i=0;i<n;i++){
        if(x[i]>0.0f){
            y[i]=x[i];
        }else{
            y[i]=0.0f;
        }
    }
}
/*
ReLUを要素ごとに計算する。

*/
void softmax(int n,const float *x,float *y){
    float max_value=x[0];
    for(int i=0;i<n;i++){
        if(x[i]>max_value){
            max_value=x[i];
        }
    }

    float sum=0.0f;
    for(int i=0;i<n;i++){
        sum+=expf(x[i]-max_value);
    }
    for(int i=0;i<n;i++){
        y[i]= expf(x[i] - max_value) / sum;
    }
}

int argmax(int n,const float *values){
    int max_index=0;
    for(int i=1;i<n;i++){
        if(values[i]>values[max_index]){
            max_index=i;
        }
    }
    return max_index;
}
/*6層NNの出力をy[0]~y[9]に書き込む*/
void forward6(
    const float *A1, const float *b1,
    const float *A2, const float *b2,
    const float *A3, const float *b3,
    const float *x,
    float *y
)
{
    float u1[50];
    float z1[50];
    float u2[100];
    float z2[100];
    float u3[10];
    fc(50, 784, x, A1, b1, u1);
    relu(50, u1, z1);
    fc(100, 50, z1, A2, b2, u2);
    relu(100, u2, z2);
    fc(10, 100, z2, A3, b3, u3);
    softmax(10, u3, y);
}

/*
6層NNの学習用forward。
逆伝播で使う途中結果 u1,z1,u2,z2,u3,y をすべて保存する。
*/
void forward6_train(
    const float *A1, const float *b1,
    const float *A2, const float *b2,
    const float *A3, const float *b3,
    const float *x,
    float *u1, float *z1,
    float *u2, float *z2,
    float *u3, float *y
)
{
    fc(50, 784, x, A1, b1, u1);
    relu(50, u1, z1);
    fc(100, 50, z1, A2, b2, u2);
    relu(100, u2, z2);
    fc(10, 100, z2, A3, b3, u3);
    softmax(10, u3, y);
}

/*正解tに対する交差エントロピー誤差を返す*/
float cross_entropy_error(const float *y,int t){
    const float epsilon=1.0e-9f;
    return -logf(y[t]+epsilon);
}

/*
Softmax+Cross Entropyの逆伝播
yはSoftmaxの出力、tは正解ラベル0~9
dEdx=y-tはSoftmax入力に対する勾配
*/
void softmaxwithloss_bwd(int n,const float *y,unsigned char t,float *dEdx){
    for(int i=0;i<n;i++){
        dEdx[i]=y[i];
    }
    dEdx[t]-=1.0f;
}

/*
ReLUの逆伝播
xは逆伝播時の、ReLUへの入力
dEdyは上から来た勾配
dEdxはReLU入力に対する勾配
*/
void relu_bwd(int n,const float *x,const float *dEdy,float *dEdx){
    for(int i=0;i<n;i++){
        if(x[i]>0){
            dEdx[i]=dEdy[i];
        }else{
            dEdx[i]=0.0f;
        }
    }
}

/*
FC層 y = Ax + b の逆伝播。
m: 出力次元
n: 入力次元
x: 順伝播時のFC入力
A: 順伝播時に使った行列
dEdy: 上流から来た勾配 dE/dy
dEdA: Aに対する勾配を書き込む
dEdb: bに対する勾配を書き込む
dEdx: xに対する勾配を書き込む。
*/
void fc_bwd(int m,int n,const float *x,const float *dEdy,const float *A,float *dEdA,float *dEdb,float *dEdx){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            dEdA[i*n+j]=dEdy[i]*x[j];
        }
    }
    for(int i=0;i<m;i++){
        dEdb[i]=dEdy[i];
    }
    if (dEdx != NULL) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int i = 0; i < m; i++) {
                sum += A[i * n + j] * dEdy[i];
            }
            dEdx[j] = sum;
        }
    }
}

/*
6層NNの逆伝播。
1枚の画像 x と正解 t から、各パラメータの勾配を計算する。
y には Softmax出力を書き込む。
*/
void backward6(
    const float *A1, const float *b1,
    const float *A2, const float *b2,
    const float *A3, const float *b3,
    const float *x,
    unsigned char t,
    float *y,
    float *dEdA1, float *dEdb1,
    float *dEdA2, float *dEdb2,
    float *dEdA3, float *dEdb3
)
{
    float u1[50];
    float z1[50];
    float u2[100];
    float z2[100];
    float u3[10];
    float dEd_u3[10];
    float dEd_z2[100];
    float dEd_u2[100];
    float dEd_z1[50];
    float dEd_u1[50];

    forward6_train(A1, b1, A2, b2, A3, b3, x,
                    u1, z1, u2, z2, u3, y);
    /* Softmax + Cross Entropy */
    softmaxwithloss_bwd(10, y, t, dEd_u3);
    /* FC3: z2 → u3 */
    fc_bwd(10, 100, z2, dEd_u3, A3,
            dEdA3, dEdb3, dEd_z2);
    /* ReLU2: u2 → z2 */
    relu_bwd(100, u2, dEd_z2, dEd_u2);
    /* FC2: z1 → u2 */
    fc_bwd(100, 50, z1, dEd_u2, A2,
            dEdA2, dEdb2, dEd_z1);
    /* ReLU1: u1 →　z1 */
    relu_bwd(50, u1, dEd_z1, dEd_u1);
    /* FC1: x → u1 */
    fc_bwd(50, 784, x, dEd_u1, A1,
            dEdA1, dEdb1, NULL);
}
/*
SGDで使う関数
*/

/*配列o[0]~o[n-1]をすべてxで初期化*/
void init(int n,float x,float *o){
    for(int i=0;i<n;i++){
        o[i]=x;
    }
}

/*o[i]+=x[i]*/
void add(int n,const float *x,float *o){
    for(int i=0;i<n;i++){
        o[i]+=x[i];
    }
}

/*o[i]*=x*/
void scale(int n,float x,float *o){
    for(int i=0;i<n;i++){
        o[i]*=x;
    }
}

/*
o[i]を[-1,1]の乱数で初期化する
改善後に不要になった。
*/
void rand_init(int n,float *o){
    for(int i=0;i<n;i++){
        float r=(float)rand()/(float)RAND_MAX;
        o[i]=2.0f*r-1.0f;
    }
}

void rand_init_fc(int m, int n, float *A)
{
    float limit = sqrtf(6.0f / (float)n);

    for (int i = 0; i < m * n; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        A[i] = (2.0f * r - 1.0f) * limit;
    }
}

/*長さnの配列xをランダムに並び替える*/
void shuffle(int n,int *x){
    for(int i=0;i<n;i++){
        int j=rand()%n;
        int tmp=x[i];
        x[i]=x[j];
        x[j]=tmp;
    }
}

void save(const char *filename, int m, int n, const float *A, const float *b)
{
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
        fprintf(stderr, "cannot open %s for write\n", filename);
        exit(1);
    }
    if (fwrite(A, sizeof(float), m * n, fp) != (size_t)(m * n)) {
        fprintf(stderr, "failed to write A to %s\n", filename);
        fclose(fp);
        exit(1);
    }
    if (fwrite(b, sizeof(float), m, fp) != (size_t)m) {
        fprintf(stderr, "failed to write b to %s\n", filename);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
}
void load(const char *filename, int m, int n, float *A, float *b)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "cannot open %s for read\n", filename);
        exit(1);
    }
    if (fread(A, sizeof(float), m * n, fp) != (size_t)(m * n)) {
        fprintf(stderr, "failed to read A from %s\n", filename);
        fclose(fp);
        exit(1);
    }
    if (fread(b, sizeof(float), m, fp) != (size_t)m) {
        fprintf(stderr, "failed to read b from %s\n", filename);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
}

/*
現在のA,bで指定されたデータ全体の平均損失と正解率を表示する。
name   : 表示用の名前。例 "train", "test"
A, b   : 3層NNのパラメータ
images : 画像データ。1枚あたり 784 個の float
labels : 正解ラベル。0〜9
count  : 画像枚数
*/
/*
現在の6層NNパラメータで、データ全体の平均損失と正解率を表示する。
*/
void evaluate6(
    const char *name,
    const float *A1, const float *b1,
    const float *A2, const float *b2,
    const float *A3, const float *b3,
    const float *images,
    const unsigned char *labels,
    int count
)
{
    float y[10];
    double total_loss = 0.0;
    int correct_count = 0;

    for (int i = 0; i < count; i++) {
        const float *image = images + 784 * i;
        int correct_label = labels[i];
        forward6(A1, b1, A2, b2, A3, b3, image, y);
        total_loss += cross_entropy_error(y, correct_label);
        if (argmax(10, y) == correct_label) {
            correct_count++;
        }
    }

    printf("%s loss = %.6f, accuracy = %.2f%%\n",name,total_loss / count,100.0 * correct_count / count);
}

void train6(
float *A1, float *b1,
float *A2, float *b2,
float *A3, float *b3,
const float *train_x,
const unsigned char *train_y,
int train_count,
const float *test_x,
const unsigned char *test_y,
int test_count
)
{
const int epoch_count = 20;
const int batch_size = 100;
const float learning_rate = 0.01f;
int *index = malloc(sizeof(int) * train_count);
float *dEdA1_tmp = malloc(sizeof(float) * 50 * 784);
float *dEdA1_sum = malloc(sizeof(float) * 50 * 784);
float *dEdA2_tmp = malloc(sizeof(float) * 100 * 50);
float *dEdA2_sum = malloc(sizeof(float) * 100 * 50);
float *dEdA3_tmp = malloc(sizeof(float) * 10 * 100);
float *dEdA3_sum = malloc(sizeof(float) * 10 * 100);
if (index == NULL || dEdA1_tmp == NULL || dEdA1_sum == NULL ||
dEdA2_tmp == NULL || dEdA2_sum == NULL ||
dEdA3_tmp == NULL || dEdA3_sum == NULL) {
fprintf(stderr, "malloc failed in train6\n");
free(index);
free(dEdA1_tmp); free(dEdA1_sum);
free(dEdA2_tmp); free(dEdA2_sum);
free(dEdA3_tmp); free(dEdA3_sum);
return;
}
float y[10];
float dEdb1_tmp[50], dEdb1_sum[50];
float dEdb2_tmp[100], dEdb2_sum[100];
float dEdb3_tmp[10], dEdb3_sum[10];
for (int i = 0; i < train_count; i++) {
index[i] = i;
}



    for (int epoch = 0; epoch < epoch_count; epoch++) {
        shuffle(train_count, index);
        int batch_count = train_count / batch_size;
        for (int batch = 0; batch < batch_count; batch++) {
            init(50 * 784, 0.0f, dEdA1_sum);
            init(50, 0.0f, dEdb1_sum);
            init(100 * 50, 0.0f, dEdA2_sum);
            init(100, 0.0f, dEdb2_sum);
            init(10 * 100, 0.0f, dEdA3_sum);
            init(10, 0.0f, dEdb3_sum);
            for (int k = 0; k < batch_size; k++) {
                int data_index = index[batch * batch_size + k];
                const float *image = train_x + 784 * data_index;
                unsigned char label = train_y[data_index];
                backward6(A1, b1, A2, b2, A3, b3,
                            image, label, y,
                            dEdA1_tmp, dEdb1_tmp,
                            dEdA2_tmp, dEdb2_tmp,
                            dEdA3_tmp, dEdb3_tmp);
                add(50 * 784, dEdA1_tmp, dEdA1_sum);
                add(50, dEdb1_tmp, dEdb1_sum);
                add(100 * 50, dEdA2_tmp, dEdA2_sum);
                add(100, dEdb2_tmp, dEdb2_sum);
                add(10 * 100, dEdA3_tmp, dEdA3_sum);
                add(10, dEdb3_tmp, dEdb3_sum);
            }
            scale(50 * 784, -learning_rate / batch_size, dEdA1_sum);
            scale(50, -learning_rate / batch_size, dEdb1_sum);
            scale(100 * 50, -learning_rate / batch_size, dEdA2_sum);
            scale(100, -learning_rate / batch_size, dEdb2_sum);
            scale(10 * 100, -learning_rate / batch_size, dEdA3_sum);
            scale(10, -learning_rate / batch_size, dEdb3_sum);
            add(50 * 784, dEdA1_sum, A1);
            add(50, dEdb1_sum, b1);
            add(100 * 50, dEdA2_sum, A2);
            add(100, dEdb2_sum, b2);
            add(10 * 100, dEdA3_sum, A3);
            add(10, dEdb3_sum, b3);
        }
        printf("epoch %d\n", epoch + 1);
        evaluate6("train", A1, b1, A2, b2, A3, b3,
                    train_x, train_y, train_count);
        evaluate6("test ", A1, b1, A2, b2, A3, b3,
                    test_x, test_y, test_count);
    }
    free(index);
    free(dEdA1_tmp); free(dEdA1_sum);
    free(dEdA2_tmp); free(dEdA2_sum);
    free(dEdA3_tmp); free(dEdA3_sum);
}

