#include <cstdio>
#include <cstdlib>
// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

template <typename T>
void alloc_bytes(T &ptr, size_t num_bytes){

  cudaMallocManaged(&ptr, num_bytes);
}

__global__ void inc(int *array, size_t n){
  size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
  while (idx < n){
    array[idx]++;
    idx += blockDim.x*gridDim.x; // grid-stride loop
    }
}

const size_t  ds = 32ULL*1024ULL*1024ULL;

int main(){

//  int nDevices;
//   cudaGetDeviceCount(&nDevices);
  
//   printf("Number of devices: %d\n", nDevices);
  
//   for (int i = 0; i < nDevices; i++) {
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, i);
//     printf("Device Number: %d\n", i);
//     printf("  Device name: %s\n", prop.name);
//     printf("  Memory Clock Rate (MHz): %d\n",
//            prop.memoryClockRate/1024);
//     printf("  Memory Bus Width (bits): %d\n",
//            prop.memoryBusWidth);
//     printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
//            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
//     printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
//     printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
//     printf("  minor-major: %d-%d\n", prop.minor, prop.major);
//     printf("  Warp-size: %d\n", prop.warpSize);
//     printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
//     printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
//   }

  int *h_array;
  alloc_bytes(h_array, ds*sizeof(h_array[0]));
  cudaCheckErrors("cudaMallocManaged Error");
  memset(h_array, 0, ds*sizeof(h_array[0]));
  // cudaMemPrefetchAsync(h_array, ds*sizeof(h_array[0]), 0); // add in step 2c
  // cudaCheckErrors("cudaMemPrefetchAsync error");
  inc<<<256, 256>>>(h_array, ds);
  cudaCheckErrors("kernel launch error");
  // cudaMemPrefetchAsync(h_array, ds*sizeof(h_array[0]), cudaCpuDeviceId); // add in step 2c
  cudaDeviceSynchronize();
  cudaCheckErrors("kernel execution error");
  for (int i = 0; i < ds; i++) 
    if (h_array[i] != 1) {printf("mismatch at %d, was: %d, expected: %d\n", i, h_array[i], 1); return -1;}
  printf("success!\n"); 
  return 0;
}
