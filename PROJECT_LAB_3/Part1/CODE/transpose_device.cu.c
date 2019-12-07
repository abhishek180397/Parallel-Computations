#include <cassert>
#include <cuda_runtime.h>
#include "transpose_cuda.cuh"

/**
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
*/


/**
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304  matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {

  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  const int end_j = j + 4;

  /*Each submatrix of dimension 32 X 4 is handled by one warp and each column 
  is handled by a thread so that a warp reads one row per instruction from 
  32 columns. Hence read is coalesced by considering that input matrix is in row major format.
  But as each warp will be writing to 32 consecutive rows, the write is not coalesced. 
  As a result, a warp writes to 32 different 128 byte cache lines.

  In order to store the 64 X 64 matrix per block, we will use shared memory in 
  which a warp reads in a row and writes as a column to shared memory. The read
  is coalesced as we want to read a row at a time and in order to avoid bank 
  conflicts, we do padding to the shared memory so that a warp reads from shared 
  memory a d writes to a row to output using the transposed indices which are correct. 
  Then we will be writing into the row again so that the write is coalesced.*/

  for (; j < end_j; j++) {
    output[j + n * i] = input[i + n * j];
  }
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {

  /* We will be able to avoid bank conflicts since we are accessing the shared 
  memory stride 65. Submatrix of dimension 64 X 64 will be stored in shared 
  memory and is padded by a column at the end.*/

  __shared__ float data[65*64];

  // Reading from input
  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  const int end_j = j + 4;

  // Writing to output
  const int i_t = threadIdx.x + 64 * blockIdx.y;
  int j_t = 4 * threadIdx.y + 64 * blockIdx.x;
  const int end_j_t = j_t + 4;

  // Reading/writing to shared memory
  const int i_data = threadIdx.x;
  int j_data = 4 * threadIdx.y;

  /* Reading from input is coalesced as every warp reads from 32 
  consecutive columns as 1 row per instruction. A warp reads and 
  writes as a column in 1 instruction. By padding the shared memory 
  we are removing bank conflicts as there will be no threads sharing 
  the same bank.*/

  for (; j < end_j; j++) {
    data[j_data + 65*i_data] = input[i + n * j];
    j_data++;
  }
  __syncthreads();
  j_data -= 4;
  
  /*There are no bank conflicts now as each warp reads a row from 
  shared memory. By using the transposed indices, we will write a 
  row as a row into output and the write is coalesced as a warp writes 
  32 consecutive columns as one row per one instruction.*/

  for (; j_t < end_j_t; j_t++) {
    output[i_t + n * j_t] = data[i_data + 65 * j_data];
    j_data++;
  }

}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
  __shared__ float data[65*64];

  /* All of these could be performed at the same time as they are not dependent on each other.*/

  const int i = threadIdx.x + 64 * blockIdx.x;
  const int j = 4 * threadIdx.y + 64 * blockIdx.y;

  const int i_data = threadIdx.x;
  const int j_data = 4 * threadIdx.y;

  const int i_t = threadIdx.x + 64 * blockIdx.y;
  const int j_t = 4 * threadIdx.y + 64 * blockIdx.x;

  // Loop unrolling
  data[j_data + 65*i_data] = input[i + n * j];
  data[j_data + 1 + 65*i_data] = input[i + n * (j+1)];
  data[j_data + 2 + 65*i_data] = input[i + n * (j+2)];
  data[j_data + 3 + 65*i_data] = input[i + n * (j+3)];
  __syncthreads();

  // Loop unrolling
  output[i_t + n * j_t] = data[i_data+ 65 * j_data];
  output[i_t + n * (j_t+1)] = data[i_data+ 65 * (j_data+1)];
  output[i_t + n * (j_t+2)] = data[i_data+ 65 * (j_data+2)];
  output[i_t + n * (j_t+3)] = data[i_data+ 65 * (j_data+3)];
}

void cudaTranspose(const float *d_input,
                   float *d_output,
                   int n,
                   TransposeImplementation type) {
  if (type == NAIVE) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else if (type == SHMEM) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else if (type == OPTIMAL) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else {
    // unknown type
    assert(false);
  }
}