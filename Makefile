CUDA_HOME?=/usr/local/cuda
SHMEM_HOME?=/home/yuxinc/yuxinchenPSG_Home/nvshmem_0.1.0+cuda9_x86_64

CC=gcc
CUDACC=${CUDA_HOME}/bin/nvcc

CUDACFLAGS=-c -dc -gencode arch=compute_70,code=sm_70 -Xptxas="-v" --expt-extended-lambda --std=c++11 -I${SHMEM_HOME}/include
LDFLAGS =-gencode=arch=compute_70,code=sm_70 -L$(SHMEM_HOME)/lib -lshmem -lcuda

OBJ=irregular_get.o 

all: ${OBJ}
	${CUDACC} -o irregular_get ${OBJ} ${LDFLAGS}

%.o: %.cu
	${CUDACC} ${CUDACFLAGS} $<

clean:
	rm -rf *.o irregular_get
