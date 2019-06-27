#include <iostream>

#include <cuda.h>

#include <shmem.h>

#include "time.cuh"


#define TID (threadIdx.x+blockIdx.x*blockDim.x)
#define WARPID ((threadIdx.x+blockIdx.x*blockDim.x)>>5)


__global__ void full_band(size_t size, int *remote_buffer, int *local_buffer, int remote_pe)
{
  for(uint32_t i = TID; i<size; i+=blockDim.x*gridDim.x)
  {
//    int item = shmem_int_g(remote_buffer+i, remote_pe);
//    local_buffer[i] = item;
    shmem_int_get(local_buffer+i, remote_buffer+i, 1, remote_pe);
  }
}

__global__ void full_band_warp(size_t size, int *remote_buffer, int *local_buffer, int remote_pe)
{
  for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
  {
    uint32_t i = __shfl_sync(0xffffffff, i, 0);
    __syncwarp();
    shmemx_get_int_warp(local_buffer+i, remote_buffer+i, 32, remote_pe);
  }
}
__global__ void full_band_block(size_t size, int *remote_buffer, int *local_buffer, int remote_pe)
{
  for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
  {
    shmemx_get_int_block(local_buffer+blockIdx.x*blockDim.x, remote_buffer+blockIdx.x*blockDim.x, blockDim.x, remote_pe);
  }
}

__global__ void char_band(size_t size, int *remote_buffer, int *local_buffer, int remote_pe)
{
  for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
  {
    shmem_getmem((void *)(local_buffer+i), (void *)(remote_buffer+i), sizeof(int), remote_pe);
  }
}



int main()
{
  size_t size = 1<<20;
  shmem_init();
  int my_pe = shmem_my_pe();
  int n_pe = shmem_n_pes();

  int dev_count;
  cudaGetDeviceCount(&dev_count);
  cudaSetDevice(my_pe);

  int * remote_buffer = (int *)shmem_malloc(sizeof(int)*size);
  int * local_buffer;
  cudaMallocManaged(&local_buffer, sizeof(int)*size);
  int remote_pe = my_pe^1;

  GpuTimer timer;
  float totaltime = 0.0;

  for(int i=0; i<200; i++)
  {
    timer.Start();
    char_band<<<320, 512>>>(size, remote_buffer, local_buffer, remote_pe);
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  shmem_barrier_all();
  shmem_finalize();
  totaltime = totaltime/200;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
  std::cout << "end of the program: "<< my_pe << std::endl;
  return 0;
}
