#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.hpp>

#include <stdio.h>
#include <unistd.h>

static cublasHandle_t g_cublas;


static void
checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    exit(1);
  }
}


static void
testmm_fp16(int size, const void *a, const void *b, void *c)
{
    __half halpha = 1.0f;
    __half hbeta = 0.0f;

    int r = cublasHgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        size, size, size,
                        &halpha, (const __half *)a, size,
                        (const __half *)b, size, &hbeta,
                        (__half *)c, size);
    if(r) {
        fprintf(stderr, "cublas failed\n");
        exit(2);
    }
}

static void
testmm_fp32(int size, const void *a, const void *b, void *c)
{
    float halpha = 1.0f;
    float hbeta = 0.0f;

    int r = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        size, size, size,
                        &halpha, (const float *)a, size,
                        (const float *)b, size, &hbeta,
                        (float *)c, size);
    if(r) {
        fprintf(stderr, "cublas failed\n");
        exit(2);
    }
}

template <typename T>
static void
fillrand(T *a, size_t cnt)
{
    for(size_t i = 0; i < cnt; i++) {
        a[i] = drand48();
    }
}

int
main(int argc, char **argv)
{
    int rounds = 100;
    size_t size = 8192;
    int half = 0;
    int opt;

    int use_tensorcore = 0;
    while ((opt = getopt(argc, argv, "ths:r:")) != -1) {
        switch (opt) {
        case 'h':
            half = 1;
            break;
        case 't':
            use_tensorcore = 1;
            break;
        case 's':
            size = atoi(optarg);
            break;
        case 'r':
            rounds = atoi(optarg);
            break;
        }
    }

    size_t s2 = size * size;
    size_t es = half ? 2 : 4;
    void *a, *b, *c;

    cublasCreate(&g_cublas);
    if(!use_tensorcore)
        cublasSetMathMode(g_cublas, CUBLAS_PEDANTIC_MATH);

    checkCuda(cudaMallocManaged(&a, s2 * es));
    checkCuda(cudaMallocManaged(&b, s2 * es));
    checkCuda(cudaMallocManaged(&c, s2 * es));

    if(half) {
        fillrand((__half *)a, s2);
        fillrand((__half *)b, s2);
    } else {
        fillrand((float *)a, s2);
        fillrand((float *)b, s2);
    }

    printf("Starting\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float sum = 0;
    for(int i = 0; i < rounds; i++) {
        cudaEventRecord(start, 0);
        if(half) {
            testmm_fp16(size, a, b, c);
        } else {
            testmm_fp32(size, a, b, c);
        }
        if(!i)
            continue; // Don't account first round
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        sum += elapsed;
    }
    printf("%f seconds\n", (sum / rounds-1) / 1000.0f);
}
