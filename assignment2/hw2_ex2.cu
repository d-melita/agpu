#include <stdio.h>
#include <sys/time.h>

#define DataType double
#define TPB 32

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < numARows && col < numBColumns) {
    DataType value = 0;
    for (int k = 0; k < numAColumns; k++) {
      value += A[row * numAColumns + k] * B[k * numBColumns + col];
    }
    C[row * numBColumns + col] = value;
  }
}

// Function to get the current time in seconds
double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  numARows =  atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = numAColumns;
  numBColumns = atoi(argv[3]);
  numCRows = numARows;
  numCColumns = numBColumns;

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  // Allocate Host memory
  hostA = (DataType*)malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType*)malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType*)malloc(numCRows * numCColumns * sizeof(DataType));
  resultRef = (DataType*)malloc(numCRows * numCColumns * sizeof(DataType));
  
  // Initialize hostA and hostB to random numbers
  for (int i = 0; i < numARows * numAColumns; i++) {
      hostA[i] = rand()/(DataType)RAND_MAX;
  }

  for (int i = 0; i < numBRows * numBColumns; i++) {
      hostB[i] = rand()/(DataType)RAND_MAX;
  }

  // Create reference result in CPU
  for (int i = 0; i < numARows; i++) {
    for (int j = 0; j < numBColumns; j++) {
      DataType value = 0;
      for (int k = 0; k < numAColumns; k++) {
        value += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
      }
      resultRef[i * numBColumns + j] = value;
    }
  }

  // Allocate GPU memory
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(DataType));

  // Timer for Host to Device memory copy
  double start_time = cpuSecond();
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  double hostToDeviceTime = cpuSecond() - start_time;
  printf("Time for host to device memory copy: %f seconds\n", hostToDeviceTime);

  // Initialize the grid and block dimensions
  dim3 blockSize(TPB, TPB);
  dim3 gridSize((numCColumns + TPB - 1) / TPB, (numCRows + TPB - 1) / TPB);

  // Timer for CUDA kernel execution
  start_time = cpuSecond();
  gemm<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  double kernelTime = cpuSecond() - start_time;
  printf("Time for kernel execution: %f seconds\n", kernelTime);

  // Timer for Device to Host memory copy
  start_time = cpuSecond();
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
  double deviceToHostTime = cpuSecond() - start_time;
  printf("Time for device to host memory copy: %f seconds\n", deviceToHostTime);

  // Compare the output with the reference
  bool match = true;
  for (int i = 0; i < numCRows * numCColumns; i++) {
    if (fabs(hostC[i] - resultRef[i]) > 1e-6) {
      match = false;
      printf("Mismatch at index %d: hostOutput[%d] = %f, resultRef[%d] = %f\n", i, i, hostC[i], i, resultRef[i]);
      break;
    }
  }
  if (match) {
    printf("Test PASSED\n");
  } else {
    printf("Test FAILED\n");
  }

  // Free the GPU memory
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  // Free the CPU memory
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}
