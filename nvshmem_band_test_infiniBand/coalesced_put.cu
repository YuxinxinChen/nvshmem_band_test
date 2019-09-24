/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#include <stdio.h>
#include <iostream>
#include "mpi.h"
#include "nvshmem.h"
#include "nvshmemx.h"

#include "time.cuh"

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                \
do {                                                    \
    cudaError_t result = (stmt);                        \
    if (cudaSuccess != result) {                        \
        fprintf(stderr, "[%s:%d] cuda failed with %s \n",\
         __FILE__, __LINE__, cudaGetErrorString(result));\
        exit(-1);					\
    }                                                   \
} while (0)

#define MPI_CHECK(stmt)                                 \
do {                                                    \
    int result = (stmt);                                \
    if (MPI_SUCCESS != result) {                        \
        fprintf(stderr, "[%s:%d] MPI failed with error %d \n",\
         __FILE__, __LINE__, result);                   \
        exit(-1);					\
    }                                                   \
} while (0)

#define TID (threadIdx.x+blockIdx.x*blockDim.x)
#define WARPID ((threadIdx.x+blockIdx.x*blockDim.x)>>5)

__global__ void int_band(size_t bytes, int *remote_buffer, int *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(int);
  for(uint32_t i = TID; i<size; i+=blockDim.x*gridDim.x)
  {
    nvshmem_int_put(remote_buffer+i, local_buffer+i, 1, remote_pe);
  }
}
__global__ void int_band_warp(size_t bytes, int *remote_buffer, int *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(int);
  for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
  {
    i = __shfl_sync(0xffffffff, i, 0);
    __syncwarp();
    nvshmemx_int_put_warp(remote_buffer+i, local_buffer+i, 32, remote_pe);
  }
}
__global__ void int_band_block(size_t bytes, int *remote_buffer, int *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(int);
  for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
  {
    nvshmemx_int_put_block(remote_buffer+blockIdx.x*blockDim.x, local_buffer+blockIdx.x*blockDim.x, blockDim.x, remote_pe);
  }
}
__global__ void char_band(size_t bytes, int *remote_buffer, int *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(char);
  for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
  {
    nvshmem_putmem((void *)((char *)remote_buffer+i), (void *)((char *)local_buffer+i), sizeof(char), remote_pe);
  }
}

__global__ void char_band2(size_t bytes, int *remote_buffer, int *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(int);
  for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
  {
    nvshmem_putmem((void *)(remote_buffer+i), (void *)(local_buffer+i), sizeof(int), remote_pe);
  }
}

__global__ void long_band(size_t bytes, long long*remote_buffer, long long *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(long long );
    for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
        nvshmem_longlong_put(remote_buffer+i, local_buffer+i, 1, remote_pe);
}

__global__ void long_band_warp(size_t bytes, long long*remote_buffer, long long *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(long long );
    for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
    {
        i = __shfl_sync(0xffffffff, i, 0);
        __syncwarp();
        nvshmemx_longlong_put_warp(remote_buffer+i, local_buffer+i, 32, remote_pe);
    }
}

__global__ void long_band_block(size_t bytes, long long*remote_buffer, long long *local_buffer, int remote_pe)
{
    uint32_t size = bytes/sizeof(long long );
    for(uint32_t i=TID; i<size; i+=blockDim.x*gridDim.x)
    {
        nvshmemx_longlong_put_block(remote_buffer+blockIdx.x*blockDim.x, local_buffer+blockIdx.x*blockDim.x, blockDim.x, remote_pe);
    }
}

int main (int c, char *v[])
{
  int rank, nranks;
  MPI_Comm mpi_comm;
  nvshmemx_init_attr_t attr;
  int mype, npes;

  MPI_CHECK(MPI_Init(&c, &v));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

  mpi_comm = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr (NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  mype = nvshmem_my_pe();
  npes = nvshmem_n_pes();

  //application picks the device each PE will use
  CUDA_CHECK(cudaSetDevice(mype));

  size_t size = 1<<25;
  int * remote_buffer = (int *)nvshmem_malloc(sizeof(int)*size*2);
  int * local_buffer;
  cudaMallocManaged(&local_buffer, sizeof(int)*size*2);

  GpuTimer timer;
  float totaltime = 0.0;
  int num_round = 200;
  cudaStream_t *streams;
  streams = (cudaStream_t *)malloc(sizeof(cudaStream_t)*(npes-1));
  for(int i = 0; i<npes-1; i++ )
      cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking);
  size_t bytes = size*sizeof(int);
  int numBlock = 160;
  int numThread = 1024;

//  if(mype == 0)
//      std::cout << "\nsending "<< bytes << " bytes to all " << npes-1 << " GPUs using nvshmem_putmem(des, src, sizeof(char))\n";
//  cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)char_band);
//  nvshmem_barrier_all();
//
//  for(int i=0; i<num_round; i++)
//  {
//    int remote_pe = (mype+1)%npes;
//    timer.Start();
//    for(int j=0; j<npes-1; j++)
//    {
//    char_band<<<numBlock/(npes-1), numThread, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
//    remote_pe = (remote_pe+1) % npes;
//    }
//    cudaDeviceSynchronize();
//    timer.Stop();
//    totaltime = totaltime + timer.ElapsedMillis();
//  }
//
//  nvshmem_barrier_all();
//  totaltime = totaltime/num_round;
//  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << " threads: " << numBlock << "x"<< numThread << std::endl;
//
//  nvshmem_barrier_all();
//  if(mype == 0)
//    std::cout << std::endl;
//  nvshmem_barrier_all();
//  totaltime= 0.0;
//  for(int i=0; i<num_round; i++)
//  {
//    int remote_pe = (mype+1)%npes;
//    timer.Start();
//    for(int j=0; j<npes-1; j++)
//    {
//    char_band<<<8, 1024, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
//    remote_pe = (remote_pe+1) % npes;
//    }
//    cudaDeviceSynchronize();
//    timer.Stop();
//    totaltime = totaltime + timer.ElapsedMillis();
//  }
//
//  nvshmem_barrier_all();
//  totaltime = totaltime/num_round;
//  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
//  std::cout << 8 << "x" << 1024 <<" = "<<8*1024 << " CUDA threads to saturate the best BW\n";
//
//  //------------------------------------------------------------------------------------------//
//  nvshmem_barrier_all();
//  if(mype == 0)
//      std::cout << "\n\nsending "<< bytes << " bytes to all " << npes-1 << " GPUs using nvshmem_putmem(dst,src, sizeof(int))\n";
//  totaltime= 0.0;
//  cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)char_band2);
//  nvshmem_barrier_all();
//
//  for(int i=0; i<num_round; i++)
//  {
//    int remote_pe = (mype+1)%npes;
//    timer.Start();
//    for(int j=0; j<npes-1; j++)
//    {
//    char_band2<<<numBlock/(npes-1), numThread, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
//    remote_pe = (remote_pe+1) % npes;
//    }
//    cudaDeviceSynchronize();
//    timer.Stop();
//    totaltime = totaltime + timer.ElapsedMillis();
//  }
//
//  nvshmem_barrier_all();
//  totaltime = totaltime/num_round;
//  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << " threads: " << numBlock << "x"<< numThread << std::endl;
//
//  nvshmem_barrier_all();
//  if(mype == 0)
//      std::cout << "\n";
//  totaltime= 0.0;
//  nvshmem_barrier_all();
//
//  for(int i=0; i<num_round; i++)
//  {
//    int remote_pe = (mype+1)%npes;
//    timer.Start();
//    for(int j=0; j<npes-1; j++)
//    {
//    char_band2<<<1, 1024, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
//    remote_pe = (remote_pe+1) % npes;
//    }
//    cudaDeviceSynchronize();
//    timer.Stop();
//    totaltime = totaltime + timer.ElapsedMillis();
//  }
//
//  nvshmem_barrier_all();
//  totaltime = totaltime/num_round;
//  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
//  std::cout << 1 << "x" << 1024 <<" = "<<1*1024 << " CUDA threads to saturate the best BW\n";
//
//  //------------------------------------------------------------------------------------------//
//  nvshmem_barrier_all();
//  if(mype == 0)
//      std::cout << "\n\nsending "<< bytes << " bytes to all " << npes-1 << " GPUs using nvshmem_int_put\n";
//  totaltime= 0.0;
//  cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)int_band);
//  nvshmem_barrier_all();
//
//  for(int i=0; i<num_round; i++)
//  {
//    int remote_pe = (mype+1)%npes;
//    timer.Start();
//    for(int j=0; j<npes-1; j++)
//    {
//    int_band<<<numBlock/(npes-1), numThread, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
//    remote_pe = (remote_pe+1) % npes;
//    }
//    cudaDeviceSynchronize();
//    timer.Stop();
//    totaltime = totaltime + timer.ElapsedMillis();
//  }
//
//  nvshmem_barrier_all();
//  totaltime = totaltime/num_round;
//  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << " threads: " << numBlock << "x"<< numThread << std::endl;
//
//  nvshmem_barrier_all();
//  if(mype == 0)
//      std::cout << "\n";
//  totaltime= 0.0;
//  nvshmem_barrier_all();
//
//  for(int i=0; i<num_round; i++)
//  {
//    int remote_pe = (mype+1)%npes;
//    timer.Start();
//    for(int j=0; j<npes-1; j++)
//    {
//    int_band<<<5, 1024, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
//    remote_pe = (remote_pe+1) % npes;
//    }
//    cudaDeviceSynchronize();
//    timer.Stop();
//    totaltime = totaltime + timer.ElapsedMillis();
//  }
//
//  nvshmem_barrier_all();
//  totaltime = totaltime/num_round;
//  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
//  std::cout << 5 << "x" << 1024 <<" = "<<5*1024 << " CUDA threads to saturate the best BW\n";
//
//  //------------------------------------------------------------------------------------------//
//  nvshmem_barrier_all();
//  if(mype == 0)
//      std::cout << "\n\nsending "<< bytes << " bytes to all " << npes-1 << " GPUs using nvshmemx_int_put_warp\n";
//  totaltime= 0.0;
//  cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)int_band_warp);
//  nvshmem_barrier_all();
//
//  for(int i=0; i<num_round; i++)
//  {
//    int remote_pe = (mype+1)%npes;
//    timer.Start();
//    for(int j=0; j<npes-1; j++)
//    {
//    int_band_warp<<<numBlock/(npes-1), numThread, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
//    remote_pe = (remote_pe+1) % npes;
//    }
//    cudaDeviceSynchronize();
//    timer.Stop();
//    totaltime = totaltime + timer.ElapsedMillis();
//  }
//
//  nvshmem_barrier_all();
//  totaltime = totaltime/num_round;
//  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << " threads: " << numBlock << "x"<< numThread << std::endl;
//
//
//  nvshmem_barrier_all();
//  if(mype == 0)
//      std::cout << "\n";
//  totaltime= 0.0;
//  nvshmem_barrier_all();
//
//  for(int i=0; i<num_round; i++)
//  {
//    int remote_pe = (mype+1)%npes;
//    timer.Start();
//    for(int j=0; j<npes-1; j++)
//    {
//    int_band_warp<<<5, 1024, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
//    remote_pe = (remote_pe+1) % npes;
//    }
//    cudaDeviceSynchronize();
//    timer.Stop();
//    totaltime = totaltime + timer.ElapsedMillis();
//  }
//
//  nvshmem_barrier_all();
//  totaltime = totaltime/num_round;
//  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s"<< std::endl;
//  std::cout << 5 << "x" << 1024 <<" = "<<5*1024 << " CUDA threads to saturate the best BW\n";
//  //------------------------------------------------------------------------------------------//
//
//  nvshmem_barrier_all();
//  if(mype == 0)
//      std::cout << "\n\nsending "<< bytes << " bytes to all " << npes-1 << " GPUs using nvshmemx_int_put_block\n";
//  totaltime= 0.0;
//  cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)int_band_block);
//  nvshmem_barrier_all();
//
//  for(int i=0; i<num_round; i++)
//  {
//    int remote_pe = (mype+1)%npes;
//    timer.Start();
//    for(int j=0; j<npes-1; j++)
//    {
//    int_band_block<<<numBlock/(npes-1), numThread, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
//    remote_pe = (remote_pe+1) % npes;
//    }
//    cudaDeviceSynchronize();
//    timer.Stop();
//    totaltime = totaltime + timer.ElapsedMillis();
//  }
//
//  nvshmem_barrier_all();
//  totaltime = totaltime/num_round;
//  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << " threads: " << numBlock << "x"<< numThread << std::endl;
//
//  nvshmem_barrier_all();
//  if(mype == 0)
//      std::cout << "\n";
//  totaltime= 0.0;
//  nvshmem_barrier_all();
//
//  for(int i=0; i<num_round; i++)
//  {
//    int remote_pe = (mype+1)%npes;
//    timer.Start();
//    for(int j=0; j<npes-1; j++)
//    {
//    int_band_block<<<4, 1024, 0, streams[j]>>>(bytes, remote_buffer, local_buffer, remote_pe);
//    remote_pe = (remote_pe+1) % npes;
//    }
//    cudaDeviceSynchronize();
//    timer.Stop();
//    totaltime = totaltime + timer.ElapsedMillis();
//  }
//
//  nvshmem_barrier_all();
//  totaltime = totaltime/num_round;
//  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
//  std::cout << 4 << "x" << 1024 <<" = "<<4*1024 << " CUDA threads to saturate the best BW\n";
//
//  //------------------------------------------------------------------------------------------//
//
//  nvshmem_barrier_all();
//  if(mype == 0)
//      std::cout << "\n\nsending "<< bytes << " bytes to all " << npes-1 << " GPUs using nvshmem_longlong_put\n";
//  totaltime= 0.0;
//  cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)long_band);
//  nvshmem_barrier_all();
//
//  for(int i=0; i<num_round; i++)
//  {
//    int remote_pe = (mype+1)%npes;
//    timer.Start();
//    for(int j=0; j<npes-1; j++)
//    {
//    long_band<<<numBlock/(npes-1), numThread, 0, streams[j]>>>(bytes, (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
//    remote_pe = (remote_pe+1) % npes;
//    }
//    cudaDeviceSynchronize();
//    timer.Stop();
//    totaltime = totaltime + timer.ElapsedMillis();
//  }
//
//  nvshmem_barrier_all();
//  totaltime = totaltime/num_round;
//  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << " threads: " << numBlock << "x"<< numThread << std::endl;
//
//  nvshmem_barrier_all();
//  if(mype == 0)
//      std::cout << "\n";
//  totaltime= 0.0;
//  nvshmem_barrier_all();
//
//  for(int i=0; i<num_round; i++)
//  {
//    int remote_pe = (mype+1)%npes;
//    timer.Start();
//    for(int j=0; j<npes-1; j++)
//    {
//    long_band<<<2, 1024, 0, streams[j]>>>(bytes, (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
//    remote_pe = (remote_pe+1) % npes;
//    }
//    cudaDeviceSynchronize();
//    timer.Stop();
//    totaltime = totaltime + timer.ElapsedMillis();
//  }
//
//  nvshmem_barrier_all();
//  totaltime = totaltime/num_round;
//  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
//  std::cout << 2 << "x" << 1024 <<" = "<<2*1024 << " CUDA threads to saturate the best BW\n";
//  //------------------------------------------------------------------------------------------//
//
//  nvshmem_barrier_all();
//  if(mype == 0)
//      std::cout << "\n\nsending "<< bytes << " bytes to all " << npes-1 << " GPUs using nvshmemx_longlong_put_warp\n";
//  totaltime= 0.0;
//  cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)long_band_warp);
//  nvshmem_barrier_all();
//
//  for(int i=0; i<num_round; i++)
//  {
//    int remote_pe = (mype+1)%npes;
//    timer.Start();
//    for(int j=0; j<npes-1; j++)
//    {
//    long_band_warp<<<numBlock/(npes-1), numThread, 0, streams[j]>>>(bytes, (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
//    remote_pe = (remote_pe+1) % npes;
//    }
//    cudaDeviceSynchronize();
//    timer.Stop();
//    totaltime = totaltime + timer.ElapsedMillis();
//  }
//  
//  nvshmem_barrier_all();
//  totaltime = totaltime/num_round;
//  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << " threads: " << numBlock << "x"<< numThread << std::endl;
//
//  nvshmem_barrier_all();
//  if(mype == 0)
//      std::cout << "\n";
//  totaltime= 0.0;
//  nvshmem_barrier_all();
//
//  for(int i=0; i<num_round; i++)
//  {
//    int remote_pe = (mype+1)%npes;
//    timer.Start();
//    for(int j=0; j<npes-1; j++)
//    {
//    long_band_warp<<<2, 1024, 0, streams[j]>>>(bytes, (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
//    remote_pe = (remote_pe+1) % npes;
//    }
//    cudaDeviceSynchronize();
//    timer.Stop();
//    totaltime = totaltime + timer.ElapsedMillis();
//  }
//    nvshmem_barrier_all();
//  totaltime = totaltime/num_round;
//  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << std::endl;
//  std::cout << 2 << "x" << 1024 <<" = "<<2*1024 << " CUDA threads to saturate the best BW\n";
//  //------------------------------------------------------------------------------------------//
  nvshmem_barrier_all();
  if(mype == 0)
      std::cout << "\n\nsending "<< bytes << " bytes to all " << npes-1 << " GPUs using nvshmemx_longlong_put_block\n";
  totaltime= 0.0;
  cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)long_band_block);
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (mype+1)%npes;
    for(int j=0; j<npes-1; j++)
    {
    long_band_block<<<numBlock/(npes-1), numThread, 0, streams[j]>>>(bytes, (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % npes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }
  
  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s" << " threads: " << numBlock << "x"<< numThread << std::endl;

  nvshmem_barrier_all();
  if(mype == 0)
      std::cout << "\n";
  totaltime= 0.0;
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (mype+1)%npes;
    timer.Start();
    for(int j=0; j<npes-1; j++)
    {
    long_band_block<<<1, 1024, 0, streams[j]>>>(bytes, (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
    remote_pe = (remote_pe+1) % npes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s"<< std::endl;
  std::cout << 1 << "x" << 1024 <<" = "<<1*1024 << " CUDA threads to saturate the best BW\n";

  //------------------------------------------------------------------------------------------//
  nvshmem_barrier_all();
  totaltime = 0.0;
  if(mype==0)
  std::cout << "using cudaMemcpyAsync\n";
  nvshmem_barrier_all();

  for(int i=0; i<num_round; i++)
  {
    int remote_pe = (mype+1)%npes;
    timer.Start();
    for(int j=0; j<npes-1; j++)
    {
    long long * remote_ptr = (long long *)nvshmem_ptr(remote_buffer, remote_pe);
    cudaMemcpyAsync(remote_ptr, local_buffer, bytes, cudaMemcpyDeviceToDevice, streams[j]);
    remote_pe = (remote_pe+1) % npes;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    totaltime = totaltime + timer.ElapsedMillis();
  }

  nvshmem_barrier_all();
  totaltime = totaltime/num_round;
  std::cout <<"PE "<< mype <<  " average time: " <<  totaltime << " bandwithd: "<<(sizeof(int)*size*(npes-1)/(totaltime/1000))/(1024*1024*1024)<<" GB/s"<< std::endl;

  nvshmem_barrier_all();
  printf("[%d of %d] run complete \n", mype, npes);

  nvshmem_free(remote_buffer);
  CUDA_CHECK(cudaFree(local_buffer));

  nvshmem_finalize();
  MPI_CHECK(MPI_Finalize());
  return 0;    
}
