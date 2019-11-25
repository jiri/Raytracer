CC=/Developer/NVIDIA/CUDA-10.1/bin/nvcc

main: main.cu
	$(CC) -std=c++14 main.cu -o main -L=. -ltbb -lOpenImageDenoise.1.1.0 -ltbbmalloc --linker-options '-rpath,@executable_path'

run: main
	./main

show: run
	open smallptcuda.ppm
