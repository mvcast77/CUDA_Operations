#include <iostream>
#include <stdlib.h>
#include "timer.h"

__global__ void matmult_tiling (int * A, int * B, int * C, int Nrows, int Ncols) {
	extern __shared__ int fetched[];
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int blockSize = blockDim.y * blockDim.x;
	for (int i = 0; i < Ncols; ++i) {
		fetched[idx * Ncols + i] = A[idx * Ncols + i];
		fetched[blockSize + i * Ncols + idy] = B[i * Ncols + idy];
	}
	__syncthreads();
	if (idx == 0 && idy == 0) {
		for (int i = 0; i < 2; ++i){
			printf("[ ");
			for (int j = 0; j < blockSize; ++j){
				printf("%d ", fetched[i * blockSize + j]);
			}
			printf("]\n");
		}
	}
	__syncthreads();

	int sum = 0;
	for (int i = 0; i < Ncols; ++i) {
		int Aval = fetched[idx * Ncols + i]; 
		int Bval = fetched[i * Nrows + idy]; 
		// printf("Thread %d, %d is at %d: (%d * %d)\n", idx, idy, sum, Aval, Bval);
		sum += fetched[idx * Ncols + i] * fetched[i * Nrows + idy];
	}
	// printf("Coord %d, %d gets %d\n", idx, idy, sum);
	C[idx * Nrows + idy] = sum;
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

	populate(B, Ncols, Nrows);
	printf("Matrix B is set to: \n");
	print_mat(B, Ncols, Nrows);

	// Sequential Recording Section
	Timer t;

	matmult_seq(A,B,C,Nrows,Ncols);

	std::cout << "Time elapsed: " << t.elapsed() << "seconds" << std::endl;
	printf("Sequential C is: \n");
	print_mat(C, Nrows, Nrows);

	// Cuda Initialization Section

	int * dA, * dB, * dC;

	cudaMalloc((int **) &dA, sizeof(int) * Nrows * Ncols);
	cudaMalloc((int **) &dB, sizeof(int) * Ncols * Nrows);
	cudaMalloc((int **) &dC, sizeof(int) * Nrows * Nrows);

	cudaMemcpy(dA, A, sizeof(int) * Nrows * Ncols, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeof(int) * Ncols * Nrows, cudaMemcpyHostToDevice);

	// Kernel Invocation Section + Timer

	int threadnumx = (Ncols < 1024) ? Ncols : 1024;
	int threadnumy = (Nrows * Ncols < 1024) ? Nrows : 1024 / threadnumx;
	dim3 blockDim(threadnumy, threadnumy);
	dim3 gridDim(1,1);
	int memSize = threadnumx * threadnumy * 2 * sizeof(int);
	printf("Num rows: %d\n", threadnumy);
	printf("Num cols: %d\n", threadnumx);

	matmult_tiling<<<gridDim, blockDim, memSize>>>(dA,dB,dC,Nrows,Ncols);

	cudaMemcpy(C, dC, sizeof(int) * Nrows * Nrows, cudaMemcpyDeviceToHost);
	printf("Final Result follows:\n");
	print_mat(C, Nrows, Nrows);

	// Cleanup Section

	free(A);
	free(B);
	free(C);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	return 0;
}
