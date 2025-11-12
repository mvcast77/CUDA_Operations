#include <iostream>
#include <stdlib.h>
#include <cuda/std/atomic>
#include "timer.h"

__global__ void matT (int * A, int * C) {
	extern __shared__ int fetch[];
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int k = 0; k < blockDim.y; k += blockDim.x) {
		fetch[(threadIdx.y + k) * blockDim.y + threadIdx.x] = A[(y + k) * blockDim.y + x];
	}
	__syncthreads();

	x = blockIdx.y * blockDim.y + threadIdx.x;
	y = blockIdx.x * blockDim.x + threadIdx.y;

	for (int k = 0; k < blockDim.y; k += blockDim.x) {
		C[(y + k) * blockDim.y + x] = fetch[(threadIdx.x) * blockDim.y + (threadIdx.y + k)];
	}
}

__global__ void naivematT (int * A, int * C, int Nrows, int Ncols) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	for (int k = 0; k < Ncols; k += blockDim.x) {
		C[(idx + k) * Ncols + idy] = A[(idy + k) * Ncols + idx];
	}
}

__global__ void matcpy (int * A, int * C, int Nrows, int Ncols) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	for (int k = 0; k < Ncols; k += blockDim.x) {
		C[(idy + k) * Ncols + idx] = A[(idy + k) * Ncols + idx];
	}
}

__global__ void matmult3D (int * A, int * B, int * C, int Nrows, int Ncols) {
	int rowID = threadIdx.x;
	int colID = threadIdx.y;
	int indID = threadIdx.z;
	if (indID == 0) C[rowID * Nrows + colID] = 0;
	int product = A[rowID * Ncols + indID] * B[indID * Nrows + colID];
	atomicAdd(C + rowID * Nrows + colID, product);
}

__global__ void matmult (int * A, int * B, int * C, int Nrows, int Ncols) {
	int rowID = threadIdx.x;
	int colID = threadIdx.y;
	
	int sum = 0;
	int i, j;
	for (i = j = 0; i < Ncols; ++i, ++j) {
		// int Aval = A[rowID * Ncols + i]; 
		// int Bval = B[j * Nrows + colID]; 
		sum += A[rowID * Ncols + i] * B[j * Nrows + colID];
		// printf("Thread %d, %d is at %d: (%d * %d)\n", rowID, colID, sum, Aval, Bval);
	}
	// printf("Coord %d, %d gets %d\n", rowID, colID, sum);
	C[rowID * Nrows + colID] = sum;
}

void matmult_seq (int * A, int * B, int * C, int Nrows, int Ncols) {
	for (int i = 0; i < Nrows; ++i) {
		for (int j = 0; j < Nrows; ++j) {
			for (int k = 0; k < Ncols; ++k) {
				C[i * Nrows + j] += A[i * Ncols + k] * B[k * Nrows + j];
			}
		}
	}
}

void populate (int * mat, int Nrows, int Ncols) {
	srand(0);
	for (int i = 0; i < Nrows; ++i) {
		for (int j = 0; j < Ncols; ++j) {
			mat[i * Ncols + j] = rand() % 10;
		}
	}
}

void print_mat (int * mat, int Nrows, int Ncols) {
	for (int i = 0; i < Nrows; ++i) {
		printf("[ ");
		for (int j = 0; j < Ncols; ++j) {
			printf("%d ", mat[i * Ncols + j]);
		}
		printf("]\n");
	}
}

int main (int argc, char ** argv) {
	if (argc < 2) {
		std::cout << "Usage is ./a.out <Nrows> <Ncols>" << std::endl;
		return 1;
	}
	
	// Initialization Section
	int Nrows = atoi(argv[1]);
	int Ncols = atoi(argv[2]);

	int * A = (int *) malloc (sizeof(int) * Nrows * Ncols);
	int * B = (int *) malloc (sizeof(int) * Ncols * Nrows);
	int * C = (int *) malloc (sizeof(int) * Nrows * Nrows);
	populate(A, Nrows, Ncols);
	printf("Matrix A is set to: \n");
	print_mat(A, Nrows, Ncols);

	//populate(B, Ncols, Nrows);
	printf("Matrix B is set to: \n");
	print_mat(B, Ncols, Nrows);

	// Sequential Recording Section
	// Timer t;

	// matmult_seq(A,B,C,Nrows,Ncols);

	// std::cout << "Time elapsed: " << t.elapsed() << "seconds" << std::endl;
	// printf("Sequential C is: \n");
	// print_mat(C, Nrows, Nrows);

	// Cuda Initialization Section

	int * dA, * dB, * dC;

	cudaMalloc((int **) &dA, sizeof(int) * Nrows * Ncols);
	cudaMalloc((int **) &dB, sizeof(int) * Ncols * Nrows);
	cudaMalloc((int **) &dC, sizeof(int) * Nrows * Nrows);

	cudaMemcpy(dA, A, sizeof(int) * Nrows * Ncols, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeof(int) * Ncols * Nrows, cudaMemcpyHostToDevice);

	// Kernel Invocation Section + Timer

	int threadperdim = (Nrows * Nrows < 1024) ? Nrows : 32;
	int numBlocks = 1;
	dim3 blockDim(threadperdim,threadperdim);
	dim3 gridDim(numBlocks, numBlocks);

	Timer t0;

	matcpy<<<gridDim, blockDim>>>(dA,dB,Nrows,Ncols);
	matT<<<gridDim, blockDim, numBlocks * threadperdim * threadperdim * sizeof(int)>>>(dA,dC);

	cudaDeviceSynchronize();

	std::cout << "Time elapsed: " << t0.elapsed() << "seconds" << std::endl;

	// Results

	cudaMemcpy(B, dB, sizeof(int) * Nrows * Nrows, cudaMemcpyDeviceToHost);
	cudaMemcpy(C, dC, sizeof(int) * Nrows * Nrows, cudaMemcpyDeviceToHost);
	printf("Copied matrix is:\n");
	print_mat(B, Ncols, Nrows);
	printf("Final Result follows:\n");
	print_mat(C, Nrows, Nrows);

	free(A);
	free(B);
	free(C);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	return 0;
}
