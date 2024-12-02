#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
#define SATURATION 127
#define TPB 256

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins)
{
  // Declare shared memory for local histogram
  __shared__ unsigned int shared_bins[NUM_BINS];

  // Initialize shared memory
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  // Initialize shared bins to 0
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
  {
    shared_bins[i] = 0;
  }
  __syncthreads();

  // Compute local histogram
  for (int i = tid; i < num_elements; i += stride)
  {
    atomicAdd(&shared_bins[input[i]], 1);
  }
  __syncthreads();

  // Merge local histogram to global memory
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
  {
    atomicAdd(&bins[i], shared_bins[i]);
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  // Clean up bins that saturate at SATURATION
  for (int i = tid; i < num_bins; i += stride)
  {
    if (bins[i] > SATURATION)
    {
      bins[i] = SATURATION;
    }
  }
}

int main(int argc, char **argv)
{

  unsigned int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  // read in inputLength from args

  // Check for the correct number of arguments
  if (argc != 2) {
      printf("Usage: histogram <num elements>\n");
      exit(1);
  }

  char *endptr;
  errno = 0; // Clear errno before conversion

  // Convert input to an unsigned long
  unsigned long ul = strtoul(argv[1], &endptr, 10);

  // Check for conversion errors
  if (errno != 0 || *endptr != '\0') {
      printf("Invalid input length\n");
      exit(1);
  }

  // Check if the value is within the range of unsigned int
  if (ul > UINT_MAX) {
      printf("Input length exceeds the maximum limit of %u\n", UINT_MAX);
      exit(1);
  }

  // Assign the value to inputLength
  inputLength = (unsigned int)ul;
  printf("The input length is %u\n", inputLength);

  // allocate Host memory for input and output
  hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
  if (hostInput == NULL) {
      printf("Memory allocation for hostInput failed. Requested size: %zu bytes\n",
             inputLength * sizeof(unsigned int));
      exit(1);
  }

  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  if (hostBins == NULL) {
      printf("Memory allocation for hostBins failed.\n");
      free(hostInput);
      exit(1);
  }

  resultRef = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  if (resultRef == NULL) {
      printf("Memory allocation for resultRef failed.\n");
      free(hostInput);
      free(hostBins);
      exit(1);
  }

  // initialize hostInput to random numbers ranging from 0 to (NUM_BINS - 1)
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, NUM_BINS - 1);
  for (unsigned long long i = 0; i < inputLength; i++)
  {
    hostInput[i] = dis(gen);
  }

  // create reference result in CPU
  memset(resultRef, 0, NUM_BINS * sizeof(unsigned int));
  for (int i = 0; i < inputLength; i++)
  {
    if (resultRef[hostInput[i]] < SATURATION)
    {
      resultRef[hostInput[i]]++;
    }
  }

  // allocate GPU memory
  cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

  // Copy memory to the GPU 
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

  // initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

  // Initialize the grid and block dimensions 
  int gridSize = (inputLength + TPB - 1) / TPB;

  // Launch the GPU Kernel 
  histogram_kernel<<<gridSize, TPB>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

  // Initialize the second grid and block dimensions 
  gridSize = (NUM_BINS + TPB - 1) / TPB;

  // Launch the second GPU Kernel 
  convert_kernel<<<gridSize, TPB>>>(deviceBins, NUM_BINS);

  // Copy the GPU memory back to the CPU 
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  // compare the output with the reference
  bool correct = true;

  for (int i = 0; i < NUM_BINS; i++)
  {
    if (hostBins[i] != resultRef[i])
    {
      printf("Mismatch at bin %d: %u (computed) != %u (reference)\n",
             i, hostBins[i], resultRef[i]);
      correct = false;
      break;
    }
  }
  if (correct)
    printf("Results match!\n");

  // Free the GPU memory 
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  // Free the CPU memory 
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}
