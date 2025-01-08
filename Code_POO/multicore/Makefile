CC = gcc
FLAGS = -O3 -fopenmp
INCLUDE = -I../Source
LIBS = -L/opt/intel/oneapi/mkl/2025.0/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

# SRC_DIR = ../Source
# SRC_FILES = $(SRC_DIR)/plan.cpp $(SRC_DIR)/region.cpp $(SRC_DIR)/sparsematrix.cpp $(SRC_DIR)/optimizer.cpp 
# MAIN_FILE = main_mkl.cpp
# TARGET = adam_mkl

SRC_DIR = ../Source
SRC_FILES = $(SRC_DIR)/plan.cpp $(SRC_DIR)/region.cpp $(SRC_DIR)/sparsematrix.cpp $(SRC_DIR)/optimizer_gradient.cpp
MAIN_FILE = main_mkl.cpp
TARGET = gradient_mkl

$(TARGET): $(MAIN_FILE) $(SRC_FILES)
	$(CC) $(FLAGS) $(INCLUDE) $^ -o $@ $(LIBS)

clean:
	rm -f $(TARGET)
