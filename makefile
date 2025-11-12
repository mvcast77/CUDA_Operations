gpu: gpucheck.cu
	nvcc gpucheck.cu

vector: vector.cu
	nvcc vector.cu

reduce: reduction.cu
	nvcc -lineinfo -Xcompiler -rdynamic -arch=sm_75 reduction.cu

mat: matcpy.cu
	nvcc -arch=sm_75 matcpy.cu

mult: matmultTile.cu
	nvcc matmultTile.cu

tile: matmultRealTile2.cu
	nvcc matmultRealTile2.cu

thrust: matthrust.cu
	nvcc -G matthrust.cu

clean:
	@ rm -f a.out *.out
