#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

__global__ void testKernel(int val) {
  printf("[%d, %d]:\t\tValue is:%d\n", 
  blockIdx.y * gridDim.x + blockIdx.x,
  threadIdx.y * blockDim.x + threadIdx.x,
  val);
}

int main(int argc, char **argv) {
  int devID;
  cudaDeviceProp props;

  // Get GPU information
  cudaGetDevice(&devID);
  cudaGetDeviceProperties(&props, devID);
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", 
          devID, props.name,
          props.major, props.minor);

  printf("printf() is called. Output:\n\n");

  // Kernel configuration, where a two-dimensional grid and
  // three-dimensional blocks are configured.
  dim3 dimGrid(2, 2);
  dim3 dimBlock(2, 2);
  testKernel<<<dimGrid, dimBlock>>>(10);
  cudaDeviceSynchronize();

  return 0;
}
