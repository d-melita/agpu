#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define DataType double
#define TPB 256

// Global variable for the number of streams
#define STREAMS 4

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < len) {
    out[idx] = in1[idx] + in2[idx];
  }
}

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char **argv) {

  int inputSize;
  int segmentSize;

  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;

  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  // timing variables
  double totalTimeHostToDevice = 0;
  double totalTimeKernel = 0;
  double totalTimeDeviceToHost = 0;

  if (argc < 3) {
    fprintf(stderr, "Usage: ./hw4_ex2 <input size> <segment size>\n");
    return 1;
  }

  char *ptr;
  inputSize = strtol(argv[1], &ptr, 10);
  segmentSize = strtol(argv[2], &ptr, 10);

  if (segmentSize <= 0 || inputSize % segmentSize != 0) {
    fprintf(stderr, "Error: segment size must be positive and divide input size\n");
    return 1;
  }
  

  hostInput1 = (DataType *)malloc(inputSize * sizeof(DataType));
  hostInput2 = (DataType *)malloc(inputSize * sizeof(DataType));
  hostOutput = (DataType *)malloc(inputSize * sizeof(DataType));
  resultRef = (DataType *)malloc(inputSize * sizeof(DataType));

  for (int i = 0; i < inputSize; i++) {
    hostInput1[i] = rand() / (DataType)RAND_MAX;
    hostInput2[i] = rand() / (DataType)RAND_MAX;
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  cudaMalloc(&deviceInput1, inputSize * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputSize * sizeof(DataType));
  cudaMalloc(&deviceOutput, inputSize * sizeof(DataType));
  
  // Init streams
  cudaStream_t streams[STREAMS];
  uint streamBytes = segmentSize * sizeof(DataType);

  for (int i = 0; i < STREAMS; i++) {
      cudaStreamCreate(&streams[i]);
  }

  int numSegments = inputSize / segmentSize;
  dim3 gridSize((segmentSize + TPB - 1) / TPB);
  dim3 blockSize(TPB);

  double startHostToDevice = cpuSecond();

  for (int i = 0; i < numSegments; i++) {
      uint offset = i * segmentSize;
      cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], streamBytes,
                      cudaMemcpyHostToDevice, streams[i % STREAMS]);
      cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], streamBytes,
                      cudaMemcpyHostToDevice, streams[i % STREAMS]);
  }
  totalTimeHostToDevice += cpuSecond() - startHostToDevice;
 
  double startKernel = cpuSecond();
  for (int i = 0; i < numSegments; i++) {
      uint offset = i * segmentSize;

      vecAdd<<<gridSize, blockSize, 0, streams[i % STREAMS]>>>(&deviceInput1[offset],
        &deviceInput2[offset],
        &deviceOutput[offset],
        segmentSize);
  }
  totalTimeKernel += cpuSecond() - startKernel; 

  double startDeviceToHost = cpuSecond();
  for (int i = 0; i < numSegments; i++) {
      uint offset = i * segmentSize;
      cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset],
                      streamBytes, cudaMemcpyDeviceToHost, streams[i % STREAMS]);
  }
  totalTimeDeviceToHost += cpuSecond() - startDeviceToHost;

  // Synchronize all streams
  for (int i = 0; i < STREAMS; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  // Print the breakdown of times
  printf("Total execution time: %f seconds\n", totalTimeHostToDevice + totalTimeKernel + totalTimeDeviceToHost);
  printf("Host to Device copy: %f seconds\n", totalTimeHostToDevice);
  printf("Kernel execution: %f seconds\n", totalTimeKernel);
  printf("Device to Host copy: %f seconds\n", totalTimeDeviceToHost);

  // Verify results
  bool match = 1;
  for (int i = 0; i < inputSize; i++) {
      if (abs(hostOutput[i] - resultRef[i]) > 1e-6) {
          printf("Mismatch at index %d: hostOutput[%d] = %f, resultRef[%d] = %f\n", i, i, hostOutput[i], i, resultRef[i]);
          match = 0;
          break;
      }
  }

  if (match) {
      printf("Test PASSED\n");
  } else {
      printf("Test FAILED\n");
  }

  //@@ Destroy streams
  for (int i = 0; i < STREAMS; i++) {
      cudaStreamDestroy(streams[i]);
  }  

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);  

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);  
  return 0;
}
