CC = g++
FLAGS = -O3 -fopenmp
INCLUDE = -I../Source
LIBS = -L/opt/intel/oneapi/mkl/2025.0/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

SRC_DIR = ../Source
SRC_FILES = $(SRC_DIR)/plan.cpp $(SRC_DIR)/region.cpp $(SRC_DIR)/sparsematrix.cpp $(SRC_DIR)/utils.cpp $(SRC_DIR)/optimize_adam.cpp 
MAIN_FILE = adam_mkl.cpp
TARGET = adam_mkl

# SRC_DIR = ../Source
# SRC_FILES = $(SRC_DIR)/plan.cpp $(SRC_DIR)/region.cpp $(SRC_DIR)/sparsematrix.cpp $(SRC_DIR)/utils.cpp $(SRC_DIR)/optimize_sgd.cpp
# MAIN_FILE = gradient_mkl.cpp
# TARGET = gradient_mkl

$(TARGET): $(MAIN_FILE) $(SRC_FILES)
	$(CC) $(FLAGS) $(INCLUDE) $^ -o $@ $(LIBS)

clean:
	rm -f $(TARGET)
