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

__global__ void long_band_block(int num_messages, int message_size, long long *remote_buffer, long long *local_buffer, int remote_pe)
{
    for(int i = blockIdx.x; i<num_messages; i+=gridDim.x)
	nvshmemx_longlong_put_block(remote_buffer+message_size*i, local_buffer+message_size*i, message_size, remote_pe);
}

__global__ void long_band_warp(int num_messages, int message_size, long long *remote_buffer, long long *local_buffer, int remote_pe)
{
    for(int i = (TID>>5); i<num_messages; i+=((blockDim.x*gridDim.x)>>5))
	nvshmemx_longlong_put_warp(remote_buffer+message_size*i, local_buffer+message_size*i, message_size, remote_pe);
}

__global__ void long_band_thread(int num_messages, int message_size, long long *remote_buffer, long long *local_buffer, int remote_pe)
{
    for(int i = TID; i<num_messages; i+=blockDim.x*gridDim.x)
	nvshmem_longlong_put(remote_buffer+message_size*i, local_buffer+message_size*i, message_size, remote_pe);
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
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  printf("[%d] has %d GPUs, setDevice on GPU %d\n", mype, deviceCount, mype%deviceCount); 
  CUDA_CHECK(cudaSetDevice(mype%deviceCount));

  int bytes = 1<<30;
  char * remote_buffer = (char *)nvshmem_malloc(sizeof(char)*bytes);
  char * local_buffer;
  local_buffer = (char *)nvshmem_malloc(sizeof(char)*bytes);

  GpuTimer timer;
  float totaltime = 0.0;
  int message_bytes = 1024;
  int num_messages = bytes/message_bytes;
  cudaStream_t *streams;
  streams = (cudaStream_t *)malloc(sizeof(cudaStream_t)*(npes-1));
  for(int i=0; i<npes-1; i++)
     cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking);
  int numBlock = 160;
  int numThread = 1024;
  int num_rounds = 20;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)long_band_block));
  nvshmem_barrier_all();
  std::cout << mype << " send "<< bytes << " bytes to "<< npes-1 << " GPUs with message size(bytes) "<< message_bytes << " using nvshmem_longlong_put_block using threads: "<< numBlock << "x"<< numThread << std::endl;
  nvshmem_barrier_all();

  for(int round = 0; round < num_rounds; round++)
  {
     int remote_pe = (mype+1)%npes;
     for(int j=0; j<npes-1; j++)
     {
        timer.Start();
        long_band_block<<<numBlock, numThread, 0, streams[0]>>>(num_messages, message_bytes/sizeof(long long), (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
        remote_pe = (remote_pe+1) % npes;
     }
     cudaDeviceSynchronize();
     timer.Stop();
     totaltime = totaltime + timer.ElapsedMillis();
  }
  nvshmem_barrier_all();
  totaltime = totaltime/num_rounds;
  std::cout << "PE "<<mype << " average time: " << totaltime << " bandwidth: "<<(bytes*(npes-1)/(totaltime/1000)/(1024*1024*1024))<<" GB/s"<<std::endl;
  nvshmem_barrier_all();
  if(mype == 0)
    std::cout << "-------------------------------\n";
  nvshmem_barrier_all();
  
  totaltime = 0.0;
  numBlock = numBlock*32;
  numThread = numThread/32;
  std::cout << mype << " send "<< bytes << " bytes to "<< npes-1 << " GPUs with message size(bytes) "<< message_bytes << " using nvshmem_longlong_put_block using threads: "<< numBlock << "x"<< numThread << std::endl;
  nvshmem_barrier_all();

  for(int round = 0; round < num_rounds; round++)
  {
     int remote_pe = (mype+1)%npes;
     for(int j=0; j<npes-1; j++)
     {
        timer.Start();
        long_band_block<<<numBlock, numThread, 0, streams[0]>>>(num_messages, message_bytes/sizeof(long long), (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
        remote_pe = (remote_pe+1) % npes;
     }
     cudaDeviceSynchronize();
     timer.Stop();
     totaltime = totaltime + timer.ElapsedMillis();
  }
  nvshmem_barrier_all();
  totaltime = totaltime/num_rounds;
  std::cout << "PE "<<mype << " average time: " << totaltime << " bandwidth: "<<(bytes*(npes-1)/(totaltime/1000)/(1024*1024*1024))<<" GB/s"<<std::endl;
  nvshmem_barrier_all();
  if(mype == 0)
    std::cout << "-------------------------------\n";
  nvshmem_barrier_all();
 
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)long_band_warp));
  totaltime = 0.0;
  nvshmem_barrier_all();
  std::cout << mype << " send "<< bytes << " bytes to "<< npes-1 << " GPUs with message size(bytes) "<< message_bytes << " using nvshmem_longlong_put_warp using threads: "<< numBlock << "x"<< numThread << std::endl;
  nvshmem_barrier_all();

  for(int round = 0; round < num_rounds; round++)
  {
     int remote_pe = (mype+1)%npes;
     for(int j=0; j<npes-1; j++)
     {
        timer.Start();
        long_band_warp<<<numBlock, numThread, 0, streams[0]>>>(num_messages, message_bytes/sizeof(long long), (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
        remote_pe = (remote_pe+1) % npes;
     }
     cudaDeviceSynchronize();
     timer.Stop();
     totaltime = totaltime + timer.ElapsedMillis();
  }
  nvshmem_barrier_all();
  totaltime = totaltime/num_rounds;
  std::cout << "PE "<<mype << " average time: " << totaltime << " bandwidth: "<<(bytes*(npes-1)/(totaltime/1000)/(1024*1024*1024))<<" GB/s"<<std::endl;
  nvshmem_barrier_all();
  if(mype == 0)
    std::cout << "-------------------------------\n";
  nvshmem_barrier_all();
  
  totaltime = 0.0;
  numBlock = numBlock*32;
  numThread = numThread/32;
  std::cout << mype << " send "<< bytes << " bytes to "<< npes-1 << " GPUs with message size(bytes) "<< message_bytes << " using nvshmem_longlong_put_warp using threads: "<< numBlock << "x"<< numThread << std::endl;
  nvshmem_barrier_all();

  for(int round = 0; round < num_rounds; round++)
  {
     int remote_pe = (mype+1)%npes;
     for(int j=0; j<npes-1; j++)
     {
        timer.Start();
        long_band_warp<<<numBlock, numThread, 0, streams[0]>>>(num_messages, message_bytes/sizeof(long long), (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
        remote_pe = (remote_pe+1) % npes;
     }
     cudaDeviceSynchronize();
     timer.Stop();
     totaltime = totaltime + timer.ElapsedMillis();
  }
  nvshmem_barrier_all();
  totaltime = totaltime/num_rounds;
  std::cout << "PE "<<mype << " average time: " << totaltime << " bandwidth: "<<(bytes*(npes-1)/(totaltime/1000)/(1024*1024*1024))<<" GB/s"<<std::endl;
  nvshmem_barrier_all();
  if(mype == 0)
    std::cout << "-------------------------------\n";
  nvshmem_barrier_all();

  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&numBlock, &numThread, (void *)long_band_thread));
  totaltime = 0.0;
  nvshmem_barrier_all();
  std::cout << mype << " send "<< bytes << " bytes to "<< npes-1 << " GPUs with message size(bytes) "<< message_bytes << " using nvshmem_longlong_put_thread using threads: "<< numBlock << "x"<< numThread << std::endl;
  nvshmem_barrier_all();

  for(int round = 0; round < num_rounds; round++)
  {
     int remote_pe = (mype+1)%npes;
     for(int j=0; j<npes-1; j++)
     {
        timer.Start();
        long_band_thread<<<numBlock, numThread, 0, streams[0]>>>(num_messages, message_bytes/sizeof(long long), (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
        remote_pe = (remote_pe+1) % npes;
     }
     cudaDeviceSynchronize();
     timer.Stop();
     totaltime = totaltime + timer.ElapsedMillis();
  }
  nvshmem_barrier_all();
  totaltime = totaltime/num_rounds;
  std::cout << "PE "<<mype << " average time: " << totaltime << " bandwidth: "<<(bytes*(npes-1)/(totaltime/1000)/(1024*1024*1024))<<" GB/s"<<std::endl;
  nvshmem_barrier_all();
  if(mype == 0)
    std::cout << "-------------------------------\n";
  nvshmem_barrier_all();
  
  totaltime = 0.0;
  numBlock = numBlock*32;
  numThread = numThread/32;
  std::cout << mype << " send "<< bytes << " bytes to "<< npes-1 << " GPUs with message size(bytes) "<< message_bytes << " using nvshmem_longlong_put_thread using threads: "<< numBlock << "x"<< numThread << std::endl;
  nvshmem_barrier_all();

  for(int round = 0; round < num_rounds; round++)
  {
     int remote_pe = (mype+1)%npes;
     for(int j=0; j<npes-1; j++)
     {
        timer.Start();
        long_band_thread<<<numBlock, numThread, 0, streams[0]>>>(num_messages, message_bytes/sizeof(long long), (long long *)remote_buffer, (long long *)local_buffer, remote_pe);
        remote_pe = (remote_pe+1) % npes;
     }
     cudaDeviceSynchronize();
     timer.Stop();
     totaltime = totaltime + timer.ElapsedMillis();
  }
  nvshmem_barrier_all();
  totaltime = totaltime/num_rounds;
  std::cout << "PE "<<mype << " average time: " << totaltime << " bandwidth: "<<(bytes*(npes-1)/(totaltime/1000)/(1024*1024*1024))<<" GB/s"<<std::endl;
  nvshmem_barrier_all();
  if(mype == 0)
    std::cout << "-------------------------------\n";
  nvshmem_barrier_all();

  nvshmem_barrier_all();
  printf("[%d of %d] run complete \n", mype, npes);
 
  nvshmem_free(remote_buffer);
  nvshmem_free(local_buffer);

  nvshmem_finalize();
  MPI_CHECK(MPI_Finalize());
  return 0;    
}
