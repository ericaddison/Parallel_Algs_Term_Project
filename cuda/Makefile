CC = g++
NVCC = nvcc
CC_FLAGS = -std=c++14
NVCC_FLAGS = -arch=compute_35 -code=sm_35
LIBS = -lm
INCLUDE_DIRS = -I../cpp/src -I/opt/apps/cuda/7.0/include
LIB_DIRS =

ODIR = obj
SDIR = src
_SRC = ft_helpers.cpp fft_cuda.cpp fft_cuda_kernels.cu main.cpp
SRC = $(patsubst %,$(SDIR)/%,$(_SRC))
_OBJ1 = $(_SRC:.cu=.o)
_OBJ2 = $(_OBJ1:.cpp=.o)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ2))


all: $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ $(LIB_DIRS) $(LIBS) -o bin/fft_cuda.out

$(ODIR)/%.o: $(SDIR)/%.cpp
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) $(SDIR)/$*.cpp -c -o $@

$(ODIR)/%.o: $(SDIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $(SDIR)/$*.cu -c -o $@

submit: all
	rm -f result*
	sbatch submit

clean:
	rm -f obj/*
	rm -f bin/*
