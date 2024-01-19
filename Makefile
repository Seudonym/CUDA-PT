dev: 
	nvcc -o Main.out src/Main.cu -Isrc/include -lSDL2main -lSDL2 -diag-suppress 1866 -diag-suppress 20014
	./Main.out