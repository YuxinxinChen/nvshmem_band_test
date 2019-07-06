#include <iostream>

#include <cuda.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "time.cuh"


#define TID (threadIdx.x+blockIdx.x*blockDim.x)
#define WARPID ((threadIdx.x+blockIdx.x*blockDim.x)>>5)


__global__ void full_band(size_t size, int *remote_buffer, int *local_buffer, int remote_pe)
{
  for(uint32_t i = TID; i<size; i+=blockDim.x*gridDim.x)
  {
    nvshmem_int_put(remote_buffer+i, local_buffer+i, 1, remote_pe);
  }
}
__global__ void full_band_warp(size_t size, int *remote_buffer, int *local_buffer, int remote_pe)
{
  for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
  {
    i = __shfl_sync(0xffffffff, i, 0);
    __syncwarp();
    nvshmemx_int_put_warp(remote_buffer+i, local_buffer+i, 32, remote_pe);
  }
}
__global__ void full_band_block(size_t size, int *remote_buffer, int *local_buffer, int remote_pe)
{
  for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
  {
    nvshmemx_int_put_block(remote_buffer+blockIdx.x*blockDim.x, local_buffer+blockIdx.x*blockDim.x, blockDim.x, remote_pe);
  }
}
__global__ void char_band(size_t size, int *remote_buffer, int *local_buffer, int remote_pe)
{
  for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
  {
    nvshmem_putmem((void *)(remote_buffer+i), (void *)(local_buffer+i), sizeof(int), remote_pe);
  }
}



int main()
{
  size_t size = 1<<25;
  nvshmem_init();
  int my_pe = nvshmem_my_pe();
  int n_pes = nvshmem_n_pes();

  int dev_count;
  cudaGetDeviceCount(&dev_count);
  cudaSetDevice(my_pe);

  int * remote_buffer = (int *)nvshmem_malloc(sizeof(int)*size*2);
  int * local_buffer;
  cudaMallocManaged(&local_buffer, sizeof(int)*size*2);

  GpuTimer timer;
  float totaltime = 0.0;
  int num_round = 200;
  cudaStream_t *streams;
  streams = (cudaStream_t *)malloc(sizeof(cudaStream_t)*(n_pes-1));
  for(int i = 0; i<n_pes-1; i++ )
      cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking);
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    full_band_block<<<80, 512, 0, streams[j]>>>(size, remote_buffer, local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  nvshmem_finalize();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
  std::cout << "end of the program: "<< my_pe << std::endl;
  return 0;
}
