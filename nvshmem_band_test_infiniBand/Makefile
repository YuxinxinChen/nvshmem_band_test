# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See COPYRIGHT.txt for license information


NVSHMEM_HOME ?= /ccs/home/yuxinc/pkg/nvshmem_0.3.0 
CUDA_HOME ?= /sw/summit/cuda/10.1.105 
MPI_HOME ?= /autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-7.4.0/spectrum-mpi-10.3.0.1-20190611-mto4jwjeylfm6xdlb5yhaphpgnyzcvh7 
MPISUPPORT ?= 1
DEBUG ?= 0
VERBOSE ?= 0

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

CUDA_INC ?= $(CUDA_HOME)/include
NVCC ?= $(CUDA_HOME)/bin/nvcc

# Better define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
NVCC_GENCODE ?= -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70
NVCC_CG_GENCODE ?= -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70

TESTCUFLAGS  := -ccbin $(CXX) 

# Use addprefix so that we can specify more than one path
TESTLDFLAGS :=

TESTINC := -I$(CUDA_INC) -I$(NVSHMEM_HOME)/include  -I$(mkfile_dir)/common

ifeq ($(MPISUPPORT), 1)
MPI_LIB = -lmpi_ibm
TESTINC += -I$(MPI_HOME)/include -DENABLE_MPI_SUPPORT
endif

ifeq ($(DEBUG), 0)
TESTCUFLAGS  += -O3 
else
TESTCUFLAGS  += -O0 -g -G -lineinfo
endif

ifneq ($(VERBOSE), 0)
TESTCUFLAGS  += -lineinfo -Xptxas -v -Xcompiler -Wall,-Wextra
endif


TESTLDFLAGS += -L$(NVSHMEM_HOME)/lib -lnvshmem -lcuda -L$(CUDA_HOME)/lib64 -lcudart  
ifeq ($(MPISUPPORT), 1)
TESTLDFLAGS += -L$(MPI_HOME)/lib $(MPI_LIB)
endif
 
.PHONY : default 
default : examples

EXAMPLECUSRCFILES := coalesced_put_type.cu coalesced_put_conf.cu coalesced_put_message_size.cu

CUPERFBIN   := $(patsubst %.cu, %, $(filter %.cu, $(EXAMPLECUSRCFILES)))
CXXPERFBIN  := $(patsubst %.cpp, %, $(filter %.cpp, $(PERFCXXSRCFILES)))

$(info $(CUPERFBIN))
$(info $(CXXPERFBIN))

examples : $(CUPERFBIN) $(CXXPERFBIN)

% : %.cu
	@printf "Compiling %-25s > %-25s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCC_GENCODE) $(TESTCUFLAGS) $(TESTINC) -rdc=true $< -o $@ $(TESTLDFLAGS)

clean : 
	rm -rf $(CUPERFBIN)
