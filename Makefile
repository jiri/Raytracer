CC=/Developer/NVIDIA/CUDA-10.1/bin/nvcc

## OIDN linker flags
# -L=. -ltbb -lOpenImageDenoise.1.1.0 -ltbbmalloc --linker-options '-rpath,@executable_path'

main: main_cpu main_gpu

main_cpu: src/main.cu
	$(CC) -DUSE_CPU -std=c++14 src/main.cu -o main_cpu

main_gpu: src/main.cu
	$(CC) -std=c++14 src/main.cu --use_fast_math -o main_gpu

run: main_gpu
	./main_gpu

show: run
	open smallptcuda.ppm
