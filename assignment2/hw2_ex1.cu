#include <stdio.h>
#include <sys/time.h>

#define DataType double
#define TPB 256

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

  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  if (argc < 2) {
    fprintf(stderr, "Invalid number of arguments. Usage: ./hw2_ex1 <inputLength>");
    return 1;
  }

  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);

  hostInput1 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType *)malloc(inputLength * sizeof(DataType));
  resultRef = (DataType *)malloc(inputLength * sizeof(DataType));

  for (int i = 0; i < inputLength; i++) {
    hostInput1[i] = rand() / (DataType)RAND_MAX;
    hostInput2[i] = rand() / (DataType)RAND_MAX;
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

  // Start timing data copy from host to device
  double startCopyToDevice = cpuSecond();
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  double elapsedCopyToDevice = cpuSecond() - startCopyToDevice;

  dim3 blockSize(TPB, 1, 1);
  dim3 gridSize((inputLength + TPB - 1) / TPB, 1, 1);

  // Start timing the CUDA kernel
  double startKernel = cpuSecond();
  vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  double elapsedKernel = cpuSecond() - startKernel;

  // Start timing data copy from device to host
  double startCopyToHost = cpuSecond();
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
  double elapsedCopyToHost = cpuSecond() - startCopyToHost;

  // Print individual timings
  printf("Time for data copy from host to device: %f seconds\n", elapsedCopyToDevice);
  printf("Time for CUDA kernel execution: %f seconds\n", elapsedKernel);
  printf("Time for data copy from device to host: %f seconds\n", elapsedCopyToHost);

  bool match = true;
  for (int i = 0; i < inputLength; i++) {
    if (abs(hostOutput[i] - resultRef[i]) > 1e-6) {
      match = false;
      printf("Mismatch at index %d: hostOutput[%d] = %f, resultRef[%d] = %f\n", i, i, hostOutput[i], i, resultRef[i]);
      break;
    }
  }
  if (match) {
    printf("Test PASSED\n");
  } else {
    printf("Test FAILED\n");
  }

  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}
