#include <iostream>

#include <cuda.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "time.cuh"


#define TID (threadIdx.x+blockIdx.x*blockDim.x)
#define WARPID ((threadIdx.x+blockIdx.x*blockDim.x)>>5)


__global__ void int_band(size_t bytes, int *remote_buffer, int *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(int);
  for(uint32_t i = TID; i<size; i+=blockDim.x*gridDim.x)
  {
    nvshmem_int_get(local_buffer+i, remote_buffer+i, 1, remote_pe);
  }
}
__global__ void int_band_warp(size_t bytes, int *remote_buffer, int *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(int);
  for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
  {
    i = __shfl_sync(0xffffffff, i, 0);
    __syncwarp();
    nvshmemx_int_get_warp(local_buffer+i, remote_buffer+i, 32, remote_pe);
  }
}
__global__ void int_band_block(size_t bytes, int *remote_buffer, int *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(int);
  for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
  {
    nvshmemx_int_get_block(local_buffer+blockIdx.x*blockDim.x, remote_buffer+blockIdx.x*blockDim.x, blockDim.x, remote_pe);
  }
}
__global__ void char_band(size_t bytes, int *remote_buffer, int *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(char);
  for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
  {
    nvshmem_getmem((void *)((char *)local_buffer+i), (void *)((char *)remote_buffer+i), sizeof(char), remote_pe);
  }
}

__global__ void char_band2(size_t bytes, int *remote_buffer, int *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(int);
  for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
  {
    nvshmem_getmem((void *)(local_buffer+i), (void *)(remote_buffer+i), sizeof(int), remote_pe);
  }
}

__global__ void long_band(size_t bytes, long long*remote_buffer, long long *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(long long );
    for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
        nvshmem_longlong_get(local_buffer+i, remote_buffer+i, 1, remote_pe);
}

__global__ void long_band_warp(size_t bytes, long long*remote_buffer, long long *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(long long );
    for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
    {
        i = __shfl_sync(0xffffffff, i, 0);
        __syncwarp();
        nvshmemx_longlong_get_warp(local_buffer+i, remote_buffer+i, 32, remote_pe);
    }
}

__global__ void long_band_block(size_t bytes, long long*remote_buffer, long long *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(long long );
    for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
    {
        nvshmemx_longlong_get_block(local_buffer+blockIdx.x*blockDim.x, remote_buffer+blockIdx.x*blockDim.x, blockDim.x, remote_pe);
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
  size_t bytes = size*sizeof(int);

  //------------------------------------------------------------------------------------------//

  if(my_pe == 0)
      std::cout << "\nsending "<< bytes << " bytes to all " << n_pes-1 << " GPUs using nvshmem_getmem(des, src, sizeof(char))\n";
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    char_band<<<80, 512, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;

  nvshmem_barrier_all();
  if(my_pe == 0)
    std::cout << std::endl;
  nvshmem_barrier_all();
  totaltime= 0.0;
    for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    char_band<<<9, 1024, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
  std::cout << 9 << "x" << 1024 <<" = "<<9*1024 << " CUDA threads to saturate the best BW\n";

  //------------------------------------------------------------------------------------------//

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n\nsending "<< bytes << " bytes to all " << n_pes-1 << " GPUs using nvshmem_getmem(dst,src, sizeof(int))\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    char_band2<<<80, 512, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    char_band2<<<9, 1024, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
  std::cout << 9 << "x" << 1024 <<" = "<<9*1024 << " CUDA threads to saturate the best BW\n";

  //------------------------------------------------------------------------------------------//

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n\nsending "<< bytes << " bytes to all " << n_pes-1 << " GPUs using nvshmem_int_get\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    int_band<<<80, 512, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    int_band<<<5, 1024, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
  std::cout << 5 << "x" << 1024 <<" = "<<5*1024 << " CUDA threads to saturate the best BW\n";

  //------------------------------------------------------------------------------------------//

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n\nsending "<< bytes << " bytes to all " << n_pes-1 << " GPUs using nvshmemx_int_get_warp\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    int_band_warp<<<80, 512, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;


  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    int_band_warp<<<5, 1024, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
  std::cout << 5 << "x" << 1024 <<" = "<<5*1024 << " CUDA threads to saturate the best BW\n";

  //------------------------------------------------------------------------------------------//

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n\nsending "<< bytes << " bytes to all " << n_pes-1 << " GPUs using nvshmemx_int_get_block\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    int_band_block<<<80, 512, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    int_band_block<<<5, 1024, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
  std::cout << 5 << "x" << 1024 <<" = "<<5*1024 << " CUDA threads to saturate the best BW\n";
  //------------------------------------------------------------------------------------------//

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n\nsending "<< bytes << " bytes to all " << n_pes-1 << " GPUs using nvshmem_longlong_get\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    long_band<<<80, 512, 0, streams[j]>>>(bytes, (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    long_band<<<3, 1024, 0, streams[j]>>>(bytes, (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
  std::cout << 3 << "x" << 1024 <<" = "<<3*1024 << " CUDA threads to saturate the best BW\n";
  //------------------------------------------------------------------------------------------//

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n\nsending "<< bytes << " bytes to all " << n_pes-1 << " GPUs using nvshmemx_longlong_get_warp\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    long_band_warp<<<80, 512, 0, streams[j]>>>(bytes, (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }
  
  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    long_band_warp<<<3, 1024, 0, streams[j]>>>(bytes, (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }
    nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
  std::cout << 3 << "x" << 1024 <<" = "<<3*1024 << " CUDA threads to saturate the best BW\n";
  //------------------------------------------------------------------------------------------//

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n\nsending "<< bytes << " bytes to all " << n_pes-1 << " GPUs using nvshmemx_longlong_get_block\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    long_band_block<<<80, 512, 0, streams[j]>>>(bytes, (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }
  
  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;

  nvshmem_barrier_all();
  if(my_pe == 0)
      std::cout << "\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (my_pe+1)%n_pes;
    timer.Start();
    for(int j=0; j<n_pes-1; j++)
    {
    long_band_block<<<3, 1024, 0, streams[j]>>>(bytes, (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % n_pes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< my_pe <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(n_pes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
  std::cout << 3 << "x" << 1024 <<" = "<<3*1024 << " CUDA threads to saturate the best BW\n";

  //------------------------------------------------------------------------------------------//

  nvshmem_barrier_all();
  std::cout << "end of the program: "<< my_pe << std::endl;
  nvshmem_finalize();
  return 0;
}
