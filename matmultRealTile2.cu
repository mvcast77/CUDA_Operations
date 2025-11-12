#include <iostream>
#include <stdlib.h>
#include "timer.h"

__global__ void matmult_tiling_squared (int * A, int * B, int * C, int Nrows, int Ncols, int thread_row, int thread_col) {
	extern __shared__ int fetch[];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	for (int k = 0; k == 0 && k < gridDim.y; ++k) {
		// first, fetch sub blocks
		fetch[threadIdx.x * blockDim.y + threadIdx.y] = 
				A[threadIdx.x * Ncols + blockIdx.y * blockDim.y + threadIdx.y];
		fetch[blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x] =
				B[threadIdx.y * Nrows + blockIdx.x * blockDim.x + threadIdx.x];
		__syncthreads();
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			for (int i = 0; i < 3; ++i){
				printf("[ ");
				for (int j = 0; j < blockDim.x * blockDim.y; ++j){
					printf("%d ", fetch[i * blockDim.x * blockDim.y + j]);
				}
				printf("]\n");
			}
			printf("\n");
		}
		printf("\n");
		__syncthreads();
		// matrix multiply, add to shared
		// continue
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

	// Want to assign kernel launches for every 32 by 32 blocks
	// Number of rows covered
	int numThreadsRow = Nrows < 32 ? Nrows : 32;
	// Number of Columns covered
	int numThreadsCol = numThreadsRow;
	// Row order matrix
	dim3 blockDim(numThreadsRow,numThreadsCol);
	// How many blocks vertically?
	int numBlocksRow = (Nrows + numThreadsRow - 1) / numThreadsRow;
	// How many blocks horizontally?
	int numBlocksCol = (Ncols + numThreadsCol - 1) / numThreadsCol;
	dim3 gridDim(numBlocksRow, numBlocksCol);

	int sharedSize = numThreadsRow * numThreadsCol * 3;
	std::cout << "Num Blocks = " << numBlocksRow * numBlocksCol << std::endl;
	std::cout << "\tof size: " << numThreadsRow << " by " << numThreadsRow << std::endl;

	matmult_tiling_squared<<<gridDim, blockDim, sharedSize>>>(dA,dB,dC,Nrows,Ncols,numThreadsRow,numThreadsCol);

	cudaDeviceSynchronize();

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
