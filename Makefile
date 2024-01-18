all: 
	nvcc -o Main.out src/Main.cu -Isrc/include -diag-suppress 1866 -diag-suppress 20014
