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
  double totalTimeHostToDevice = 0;
  double totalTimeKernel = 0;
  double totalTimeDeviceToHost = 0;
  if (argc < 2) {
    fprintf(stderr, "Usage: ./hw4_ex2 <segment>\n");
    return 1;
  }

  int segment = atoi(argv[1]);
  size_t alloc_size = STREAMS * segment * sizeof(DataType);
  printf("Segment size: %d\n", segment);

  DataType *hostInput1 = (DataType *)malloc(alloc_size);
  DataType *hostInput2 = (DataType *)malloc(alloc_size);
  DataType *hostOutput = (DataType *)malloc(alloc_size);
  DataType *resultRef = (DataType *)malloc(alloc_size);

  for (int i = 0; i < segment * STREAMS; i++) {
    hostInput1[i] = rand() / (DataType)RAND_MAX;
    hostInput2[i] = rand() / (DataType)RAND_MAX;
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  // Allocate GPU memory
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  cudaMalloc(&deviceInput1, alloc_size);
  cudaMalloc(&deviceInput2, alloc_size);
  cudaMalloc(&deviceOutput, alloc_size);
  
  // Init streams

  cudaStream_t streams[STREAMS];
  uint streamBytes = segment * sizeof(DataType);

  for (int i = 0; i < STREAMS; i++) {
      cudaStreamCreate(&streams[i]);
  }

  dim3 blockSize(TPB);
  dim3 gridSize((segment + TPB - 1) / TPB);
  double startHostToDevice = cpuSecond();

  //@@ Insert code to below to Copy memory to the GPU here
  for (int i = 0; i < STREAMS; i++) {
      uint offset = i * segment;
      cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], streamBytes,
                      cudaMemcpyHostToDevice, streams[i]);
      cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], streamBytes,
                      cudaMemcpyHostToDevice, streams[i]);
  }
  totalTimeHostToDevice += cpuSecond() - startHostToDevice;
 
  double startKernel = cpuSecond();
  //@@ Launch the GPU Kernel here
  for (int i = 0; i < STREAMS; i++) {
      uint offset = i * segment;

      vecAdd<<<gridSize, blockSize, 0, streams[i]>>>(&deviceInput1[offset],
        &deviceInput2[offset],
        &deviceOutput[offset],
        segment);
  }
  totalTimeKernel += cpuSecond() - startKernel; 

  double startDeviceToHost = cpuSecond();
  //@@ Copy the GPU memory back to the CPU here
  for (int i = 0; i < STREAMS; i++) {
      uint offset = i * segment;
      cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset],
                      streamBytes, cudaMemcpyDeviceToHost, streams[i]);
  }
  totalTimeDeviceToHost += cpuSecond() - startDeviceToHost;

  // Synchronize all streams
  for (int i = 0; i < STREAMS; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  // Print the breakdown of times
  printf("Total execution time: %f seconds\n", totalTimeHostToDevice + totalTimeKernel + totalTimeDeviceToHost);
  printf("Time for Host to Device copy: %f seconds\n", totalTimeHostToDevice);
  printf("Time for Kernel execution: %f seconds\n", totalTimeKernel);
  printf("Time for Device to Host copy: %f seconds\n", totalTimeDeviceToHost);


  // Verify results
  bool match = 1;
  for (int i = 0; i < segment * STREAMS; i++) {
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
