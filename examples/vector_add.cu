#include <stdio.h>
#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <ranges>
#include <fstream>
#include <iomanip>

using namespace std;

__global__ void addVec(const float *a, const float *b, float *c, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
  {
    c[i] = a[i] + b[i];
  }
}

// Lookup opreation in cuda, UNIT test before plugging that into our actual application.

/* 
block(0,0) thread 0:
    seq      = blockIdx.x  = 0
    batch    = blockIdx.y  = 0
    e        = threadIdx.x = 0
    token_id = A[0*3 + 0]  = A[0] = 0
    val      = B[0*6 + 0]  = B[0] = 1.0  

block(0,0) thread 1:
    seq      = blockIdx.x  = 0
    batch    = blockIdx.y  = 0
    e        = threadIdx.x = 1
    token_id = A[0*3 + 0]  = A[0] = 0
    val      = B[0*6 + 1]  = B[1] = 2.0  

block(0,1) thread 0:
    seq      = blockIdx.x  = 0
    batch    = blockIdx.y  = 1
    e        = threadIdx.x = 0
    token_id = A[0*3 + 1]  = A[1] = 1
    val      = B[1*6 + 0]  = B[6] = 7.0  

*/

__global__ void embedding_lookup(
    int *A,   
    float *B,    
    float *C,  
    int embed_dim 
)
{
  int row = blockIdx.x; // seq_len dimension
  int col = blockIdx.y; // batch_size dimension
  int e = threadIdx.x;  // embed_dim dimension

  // get the token ID from A
  // A is (seq_len x batch_size)
  int token_id = A[row * gridDim.y + col];

  // get the value from B
  // B is (vocab_size x embed_dim)
  float val = B[token_id * embed_dim + e];

  // write to C
  // C is (seq_len x batch_size x embed_dim)
  // grid  = (seq_len, batch_size) y batch_size is that. 
  printf("TokenId %d val %f.1% \n", token_id, val);
  C[row * gridDim.y * embed_dim + col * embed_dim + e] = val;
}

// This is the refined unit I want to see whats going on here.
__global__ void LookUpKernel(
    int *x,           
    float *embeddings, 
    float *C,
    int d_model,
    int seq_len,
    int batch_size)
{

    int rows = blockIdx.x;
    int cols = blockIdx.y;
    int e = threadIdx.x;

    if (rows >= seq_len || cols >= batch_size)
        return;

    int index = (rows * batch_size) + cols;
    int valX = x[index]; 

    int indexB = (valX * d_model) + e; 
    float valB = embeddings[indexB];


    C[rows * batch_size * d_model + cols * d_model + e] = valB;

}


int main()
{
  int N = 10000;
  size_t size = N * sizeof(float);

  std::mt19937 gen;

  float min = 1.0f;
  float max = 5.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::uniform_real_distribution<float> dis(min, max);

  // Allocate memeory in CPU
  float *host_arr_a = (float *)malloc(size);
  float *host_arr_b = (float *)malloc(size);
  float *host_arr_c = (float *)malloc(size);

  // random values in a vector
  for (int i = 0; i < N; i++)
  {
    host_arr_a[i] = dis(gen);
    host_arr_b[i] = dis(gen);
  }

  // Allocating memory in CPU
  float *device_arr_A, *device_arr_B, *device_arr_C;
  cudaMalloc((void **)&device_arr_A, size);
  cudaMalloc((void **)&device_arr_B, size);
  cudaMalloc((void **)&device_arr_C, size);

  cudaMemcpy(host_arr_a, device_arr_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(host_arr_b, device_arr_B, size, cudaMemcpyHostToDevice);

  // kernel launch
  int threads_per_block = 256;
  int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
  cudaEventRecord(start);
  addVec<<<blocks_per_grid, threads_per_block>>>(device_arr_A, device_arr_B, device_arr_C, N);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop); // we need to wait for the GPU to stop before finishing
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("Kernel execution time: %f ms\n", milliseconds);

  cudaMemcpy(host_arr_c, device_arr_C, size, cudaMemcpyHostToDevice);

  cudaFree(device_arr_A);
  cudaFree(device_arr_B);
  cudaFree(device_arr_C);

  free(host_arr_a);
  free(host_arr_b);
  free(host_arr_c);

  // Unit test for loop up in kernel launch

  /*

  int A[9] = {
    0, 1, 2,
    3, 4, 5,
    6, 7, 8
  };


  float B[36] = {
    1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
    7.0f,  8.0f,  9.0f, 10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
    25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
    31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f
  };

  After a lookup we should expect something like

  final = [
      row 0 [1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f]
      row 1 [7.0f,  8.0f,  9.0f, 10.0f, 11.0f, 12.0f]
      row 2 [13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f]
  ]

  */
  int embed_dim = 6;

  int A[9] = {
      0, 1, 2,
      2, 1, 0,
      1, 2, 1};

  float B[36] = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
      19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
      25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
      31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f};

  // allocate GPU memory
  int *d_A;
  float *d_B;
  float *d_C;
  int seq_len = 3;
  int batch_size = 3;

  cudaMalloc(&d_A, seq_len * batch_size * sizeof(int));
  cudaMalloc(&d_B, 6 * embed_dim * sizeof(float));
  cudaMalloc(&d_C, batch_size * seq_len * embed_dim * sizeof(float));

  // copy to GPU
  cudaMemcpy(d_A, A, seq_len * batch_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, 6 * embed_dim * sizeof(float), cudaMemcpyHostToDevice);

  // launch kernel
  // grid  = (seq_len, batch_size)   each block handles one element of A
  // block = (embed_dim)             each thread handles one embed dimension
  dim3 grid(seq_len, batch_size);
  dim3 block(embed_dim);

  LookUpKernel<<<grid, block>>>(d_A, d_B, d_C, 3*3, 3, 3);

  // copy result back
  float C[seq_len * batch_size * embed_dim];
  cudaMemcpy(C, d_C, seq_len * batch_size * embed_dim * sizeof(float), cudaMemcpyDeviceToHost);

  // print result
  for (int r = 0; r < seq_len; r++)
  {
    printf("t%d:\n", r);
    for (int c = 0; c < batch_size; c++)
    {
      printf("  col%d: [", c);
      for (int e = 0; e < embed_dim; e++)
      {
        printf("%.1f ", C[r * batch_size * embed_dim + c * embed_dim + e]);
      }
      printf("]\n");
    }
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}