CC=g++
#FLAGS=-g -pg
FLAGS=-O3 -fopenmp
LIBS=-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

gradient_mkl: adam_mkl.cpp
	$(CC) $(FLAGS) $^ -o $@ $(LIBS)

clean:
	rm -f adam_mkl
