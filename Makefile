CUDA_HOME=/usr/local/cuda
SHMEM_HOME=/home/xiii/pkg/nvshmem_0.2.4-0+cuda10_x86_64

CC=gcc
CUDACC=${CUDA_HOME}/bin/nvcc

CUDACFLAGS=-c -dc -gencode arch=compute_70,code=sm_70 --expt-extended-lambda --std=c++11 -I${SHMEM_HOME}/include
LDFLAGS =-gencode=arch=compute_70,code=sm_70 -L$(SHMEM_HOME)/lib -lshmem -lcuda

OBJ=irregular_put.o 

all: ${OBJ}
	${CUDACC} -o irregular_put ${OBJ} ${LDFLAGS}

%.o: %.cu
	${CUDACC} ${CUDACFLAGS} $<

clean:
	rm -rf *.o irregular_put
