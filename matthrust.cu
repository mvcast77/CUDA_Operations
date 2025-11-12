#include <iostream>
#include <stdlib.h>
#include <thrust/universal_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include "timer.h"

__global__ void matT (int * A, int * C) {
	__shared__ int fetch[32*32];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int k = 0; k < blockDim.y; ++k) {
		fetch[(threadIdx.y + k) * blockDim.y + threadIdx.x] = A[(y + k) * blockDim.y + x];
	}
	__syncthreads();

	x = blockIdx.y * blockDim.y + threadIdx.x;
	y = blockIdx.x * blockDim.x + threadIdx.y;
	for (int k = 0; k < blockDim.y; ++k) {
		 C[(y + k) * blockDim.y + x] = 
			 fetch[(threadIdx.x) * blockDim.y + threadIdx.y + k];
	}
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

void populate (thrust::universal_vector<int>& vec, int Nrows, int Ncols) {
	srand(0);
	for (int i = 0; i < Nrows; ++i) {
		for (int j = 0; j < Ncols; ++j) {
			vec.push_back(rand() % 10);
		}
	}
}

void populate (thrust::host_vector<int>& vec, int Nrows, int Ncols) {
	srand(0);
	for (int i = 0; i < Nrows; ++i) {
		for (int j = 0; j < Ncols; ++j) {
			vec.push_back(rand() % 10);
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

void print_mat (thrust::host_vector<int>& vec, int Nrows, int Ncols) {
	for (int i = 0; i < Nrows; ++i) {
		printf("[ ");
		for (int j = 0; j < Ncols; ++j) {
			printf("%d ", vec[i * Ncols + j]);
		}
		printf("]\n");
	}
}

void print_mat (thrust::universal_vector<int>& vec, int Nrows, int Ncols) {
	for (int i = 0; i < Nrows; ++i) {
		printf("[ ");
		for (int j = 0; j < Ncols; ++j) {
			printf("%d ", vec[i * Ncols + j]);
		}
		printf("]\n");
	}
}

int main (int argc, char ** argv) {
	if (argc < 2) {
		std::cout << "Usage is ./a.out <Nrows> <Ncols>" << std::endl;
		return 1;
	}
	int Nrows = atoi(argv[1]);
	int Ncols = atoi(argv[2]);

	thrust::host_vector<int> A;
	populate(A, Nrows, Ncols);
	print_mat(A, Nrows, Ncols);

	std::cout << std::endl;

	thrust::universal_vector<int> dA = A;
	thrust::universal_vector<int> dC = A;
	// thrust::sort(dA.begin(), dA.end());
	int * iptr = thrust::raw_pointer_cast(dA.data());
	int * optr = thrust::raw_pointer_cast(dC.data());
	// cudaMemset(iptr, 0 , Nrows * Ncols);
	// cudaMemset(optr, 0 , Nrows * Ncols);
	dim3 blockDim(Nrows,Ncols);
	dim3 gridDim(1,1);

	matT<<<gridDim, blockDim>>>(iptr,optr);
	thrust::copy(dC.begin(),dC.end(),A.begin());

	print_mat(A, Nrows, Ncols);

	return 0;
}
