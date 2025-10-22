CC=nvcc
FLAGS=-O3 -std=c++14
LIBS=-lcusparse

gradient: gradient.cu
	$(CC) $(FLAGS) $^ -o $@ $(LIBS)

clean:
	rm -f gradient
