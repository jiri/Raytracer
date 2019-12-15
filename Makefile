CC=/Developer/NVIDIA/CUDA-10.1/bin/nvcc

## OIDN linker flags
# -L=. -ltbb -lOpenImageDenoise.1.1.0 -ltbbmalloc --linker-options '-rpath,@executable_path'

main: src/main.cu
	$(CC) -std=c++14 --use_fast_math src/main.cu -o main

run: main
	./main

show: run
	open smallptcuda.ppm
