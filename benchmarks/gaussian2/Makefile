include make.config

CC := g++ 
#$(CUDA_DIR)/bin/nvcc

INCLUDE := -I../../include #$(CUDA_DIR)/include

SRC = gaussian.cpp
EXE = gaus

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) $(INCLUDE) -L$(CUDA_LIB_DIR) -lcudart

clean:
	rm gaussian
