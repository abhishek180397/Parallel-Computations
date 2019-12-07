#include <cassert>
#include <cuda_runtime.h>
#include "matmul_device.cuh"

/*
 * Read TODO items below
 */




__global__
void naiveMatmul(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float acc = 0;
    for (int k=0; k<n; k++) {
	acc += a[i*n+k] * b[k*n+j];
    }
    c[i*n+j] = acc;
}

__global__
void cacheMatmul(float *a, float *b, float *c, int n) {
    // TODO: replace this function with cache friendly version
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float acc = 0;
    for (int k=0; k<n; k++) {
	acc += a[i*n+k] * b[k*n+j];
    }
    c[i*n+j] = acc;
}

__global__
void sharedMatmul(float *a, float *b, float *c, int n) {
    // TODO: replace this function with optimized code using
    // shared memory
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float acc = 0;
    for (int k=0; k<n; k++) {
	acc += a[i*n+k] * b[k*n+j];
    }
    c[i*n+j] = acc;
}

void cudaMatmul(float *a, float *b, float *c, int n, MatmulImplementation type)
{
    // TODO: play with the gridSize and blockSize to find the best one
    if (type == NAIVE) {
        dim3 blockSize(32, 32);
        dim3 gridSize(n / 32, n / 32);
        naiveMatmul<<<gridSize, blockSize>>>(a,b,c,n);
    }
    else if (type == CACHE) {
        dim3 blockSize(32, 32);
        dim3 gridSize(n / 32, n / 32);
        cacheMatmul<<<gridSize, blockSize>>>(a,b,c,n);
    }
    else if (type == SHARED) {
        dim3 blockSize(32, 32);
        dim3 gridSize(n / 32, n / 32);
        sharedMatmul<<<gridSize, blockSize>>>(a,b,c,n);
    }
    // Unknown type
    else
        assert(false);
}
