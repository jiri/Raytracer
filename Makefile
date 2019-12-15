CC=/Developer/NVIDIA/CUDA-10.1/bin/nvcc

## OIDN linker flags
# -L=. -ltbb -lOpenImageDenoise.1.1.0 -ltbbmalloc --linker-options '-rpath,@executable_path'

main: main.cu
	$(CC) -std=c++14 main.cu -o main --use_fast_math

run: main
	./main

show: run
	open smallptcuda.ppm
