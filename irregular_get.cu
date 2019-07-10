#include <iostream>
#include <cstdlib>

#include <cuda.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "time.cuh"

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
        if ( cudaSuccess != err )
                {
                            fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                                                     file, line, cudaGetErrorString( err ) );
                                    exit( -1 );
                                        }
#endif

            return;
}

#define TID (threadIdx.x+blockIdx.x*blockDim.x)
#define WARPID ((threadIdx.x+blockIdx.x*blockDim.x)>>5)


__global__ 
void int_thread(size_t size, uint32_t num_messages, uint32_t message_size, uint32_t *rand_pe, uint32_t *rand_locat, int *remote_buffer, int *local_buffer)
{
  for(uint32_t i = TID; i<num_messages; i+=blockDim.x*gridDim.x)
  {
      uint32_t location = rand_locat[i]%(size-message_size);
      nvshmem_int_get(local_buffer+TID*message_size, remote_buffer+location, message_size, rand_pe[i]);
  }
}

__global__ 
void int_warp(size_t size, uint32_t num_messages, uint32_t message_size, uint32_t *rand_pe, uint32_t *rand_locat, int *remote_buffer, int *local_buffer)
{
  for(uint32_t i = WARPID; i<num_messages; i+=((blockDim.x*gridDim.x)>>5))
  {
      uint32_t location = rand_locat[i]%(size-message_size);
      uint32_t pe = rand_pe[i];
      nvshmemx_int_get_warp(local_buffer+WARPID*message_size, remote_buffer+location, message_size, pe);
  }
}

__global__ 
void int_block(size_t size, uint32_t num_messages, uint32_t message_size, uint32_t *rand_pe, uint32_t *rand_locat, int *remote_buffer, int *local_buffer)
{
  for(uint32_t i = blockIdx.x; i<num_messages; i+=gridDim.x)
  {
      uint32_t location = rand_locat[i]%(size-message_size);
      uint32_t pe = rand_pe[i];
      nvshmemx_int_get_block( local_buffer+blockIdx.x*message_size, remote_buffer+location, message_size, pe);
  }
}

int main()
{
  size_t size = 1<<30;
  uint32_t num_messages = 1<<16; 
  uint32_t message_size[7] = {8,32, 128, 512, 2048, 8192, 32768}; //in bytes
  nvshmem_init();
  int my_pe = nvshmem_my_pe();
  int n_pes = nvshmem_n_pes();

  int dev_count;
  CudaSafeCall(cudaGetDeviceCount(&dev_count));
  CudaSafeCall(cudaSetDevice(my_pe));

  uint32_t *rand_pe;
  CudaSafeCall(cudaMallocManaged(&rand_pe, sizeof(uint32_t)*num_messages));
  uint32_t *rand_loc;
  CudaSafeCall(cudaMallocManaged(&rand_loc, sizeof(uint32_t)*num_messages));

  std::srand(12321);
  for(uint32_t i=0; i<num_messages; i++)
  {
      rand_pe[i] = std::rand()%n_pes;
      rand_loc[i] = std::rand();
  }

  int * remote_buffer = (int *)nvshmem_malloc(sizeof(int)*size);
  int * local_buffer;

  GpuTimer timer;
  float totaltime = 0.0;
  int num_round = 10;

  //------------------------------------------------------------------------------------------//

  if(my_pe == 0)
      std::cout << "\nlooking up"<< num_messages << " to all PEs with various message size using API: nvshmem_int_get\n";
  nvshmem_barrier_all();

  for(int i=0; i<7; i++)
  {
      int numBlock = 80;
      int numThread = 512;
      cudaMallocManaged(&local_buffer, message_size[i]*numBlock*numThread);
      totaltime = 0.0;
      for(int j=0; j<num_round; j++)
      {
          nvshmem_barrier_all();
          timer.Start();
          int_thread<<<80, 512>>>(size, num_messages, message_size[i]/sizeof(int), rand_pe, rand_loc, remote_buffer, local_buffer );
          cudaDeviceSynchronize();
          nvshmem_barrier_all();
          timer.Stop();
          totaltime = totaltime + timer.ElapsedMillis();
      }
      cudaFree(local_buffer);
      nvshmem_barrier_all();
      totaltime = totaltime/num_round;
      float totalGBytes = (num_messages/float(1<<30))*message_size[i];
      float bw = totalGBytes/(totaltime/1000); 
      std::cout <<"PE "<< my_pe <<" message size: "<< message_size[i]<<  " average time: " <<  totaltime << " bandwithd: "<<bw <<" GB/s\n" << std::endl;
  }

  //------------------------------------------------------------------------------------------//

  if(my_pe == 0)
      std::cout << "\nsending "<< num_messages << " to all PEs with various message size using API: nvshmemx_int_get_warp\n";
  nvshmem_barrier_all();

  for(int i=0; i<7; i++)
  {
      int numBlock = 80;
      int numThread = 512;
      cudaMallocManaged(&local_buffer, message_size[i]*numBlock*numThread/32);
      totaltime = 0.0;
      for(int j=0; j<num_round; j++)
      {
          nvshmem_barrier_all();
          timer.Start();
          int_warp<<<80, 512>>>(size, num_messages, message_size[i]/sizeof(int), rand_pe, rand_loc, remote_buffer, local_buffer );
          cudaDeviceSynchronize();
          nvshmem_barrier_all();
          timer.Stop();
          totaltime = totaltime + timer.ElapsedMillis();
      }
      cudaFree(local_buffer);
      nvshmem_barrier_all();
      totaltime = totaltime/num_round;
      float totalGBytes = (num_messages/float(1<<30))*message_size[i];
      float bw = totalGBytes/(totaltime/1000); 
      std::cout <<"PE "<< my_pe <<" message size: "<< message_size[i]<<  " average time: " <<  totaltime << " bandwithd: "<<bw <<" GB/s\n" << std::endl;
  }

  //------------------------------------------------------------------------------------------//

  if(my_pe == 0)
      std::cout << "\nsending "<< num_messages << " to all PEs with various message size using API: nvshmemx_int_get_block\n";
  nvshmem_barrier_all();

  for(int i=0; i<7; i++)
  {
      int numBlock = 80;
      int numThread = 512;
      cudaMallocManaged(&local_buffer, message_size[i]*numBlock*numThread/32);
      totaltime = 0.0;
      for(int j=0; j<num_round; j++)
      {
          nvshmem_barrier_all();
          timer.Start();
          int_block<<<80, 512>>>(size, num_messages, message_size[i]/sizeof(int), rand_pe, rand_loc, remote_buffer, local_buffer );
          cudaDeviceSynchronize();
          nvshmem_barrier_all();
          timer.Stop();
          totaltime = totaltime + timer.ElapsedMillis();
      }
      cudaFree(local_buffer);
      nvshmem_barrier_all();
      totaltime = totaltime/num_round;
      float totalGBytes = (num_messages/float(1<<30))*message_size[i];
      float bw = totalGBytes/(totaltime/1000); 
      std::cout <<"PE "<< my_pe <<" message size: "<< message_size[i]<<  " average time: " <<  totaltime << " bandwithd: "<<bw <<" GB/s\n" << std::endl;
  }

  //------------------------------------------------------------------------------------------//

  nvshmem_barrier_all();
  std::cout << "end of the program: "<< my_pe << std::endl;
  nvshmem_finalize();
  return 0;
}
