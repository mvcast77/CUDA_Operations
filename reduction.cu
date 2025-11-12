#include <iostream>
#include <stdio.h>
#include "timer.h"

__global__ void reduce0(float * g_input_data, float * g_output_data){
	extern __shared__ float sdata[];

	// Thread id within the block
	unsigned int tid = threadIdx.x;
	// Thread global id...per element in array
	unsigned int i = blockIdx.x * blockDim.x + tid;
	// printf("tid is: %d, global id is: %d\n", tid, i);
	// Each global thread is trying to put their entry in their within block spot
	// Because sdata only exists per block (declared shared)
	// So each thread is grabbing their global data and moving it per block spot
	sdata[tid] = g_input_data[i];
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) g_output_data[blockIdx.x] = sdata[0];
}

__global__ void reduce1(float * g_input_data, float * g_output_data){
	extern __shared__ float sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + tid;
	// printf("tid is: %d, global id is: %d\n", tid, i);
	sdata[tid] = g_input_data[i];
	// printf("Input is: %f\n", sdata[tid]);
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			// printf("Input for %d is: %f\n", index, sdata[index]);
			sdata[index] += sdata[index + s];
			// printf("Output for %d is: %f\n", index, sdata[index]);
		}
		__syncthreads();
	}

	// printf("Final Answer is: %f\n", sdata[0]);
	if (tid == 0) g_output_data[blockIdx.x] = sdata[0];
}

__global__ void reduce2(float * g_input_data, float * g_output_data){
	extern __shared__ float sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + tid;
	// printf("tid is: %d, global id is: %d\n", tid, i);
	sdata[tid] = g_input_data[i];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s <<= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) g_output_data[blockIdx.x] = sdata[0];
}

int main(int argc, char ** argv) {

	if (argc < 2) {
		std::cout << "Usage is ./a.out <N>" << std::endl;
		return 0;
	}
	int N = atoi(argv[1]);

	float * h_A, * h_C, * d_A, * d_C;

	h_A = (float*) malloc(N *sizeof(float));
	h_C = (float*) malloc(N *sizeof(float));

	cudaMalloc((float**) &d_A, N * sizeof(float));
	cudaMalloc((float**) &d_C, N * sizeof(float));

	for (int i = 0; i < N; ++i){
		h_A[i] = i;
		std::cout << i << ", ";
	}
	std::cout << std::endl;

	cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

	int threadperblock = (1024 > N) ? N : 1024;
	//ceil( N / tperb);
	int numBlocks = (N + threadperblock - 1) / threadperblock;
	std::cout << "Num Blocks: " << numBlocks << std::endl;
	int numElementsPerBlock = threadperblock;

	std::cout.flush();

	cudaDeviceSynchronize();

	Timer t0;
	reduce0<<<numBlocks, threadperblock, numElementsPerBlock * sizeof(float)>>>(d_A,d_C);
	cudaDeviceSynchronize();

	std::cout << "Time elapsed: " << t0.elapsed() << "seconds" << std::endl;

	cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

	fflush(stdout);

	for (int i = 0; i < N; ++i) {
		std::cout << h_C[i] << ", ";
	}
	std::cout << std::endl;

	Timer t1;
	reduce1<<<numBlocks, threadperblock, numElementsPerBlock * sizeof(float)>>>(d_A,d_C);
	cudaDeviceSynchronize();

	std::cout << "Time elapsed: " << t1.elapsed() << "seconds" << std::endl;

	cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

	fflush(stdout);

	for (int i = 0; i < N; ++i) {
		std::cout << h_C[i] << ", ";
	}
	std::cout << std::endl;

	Timer t2;
	reduce2<<<numBlocks, threadperblock, numElementsPerBlock * sizeof(float)>>>(d_A,d_C);
	cudaDeviceSynchronize();

	std::cout << "Time elapsed: " << t2.elapsed() << "seconds" << std::endl;

	cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

	fflush(stdout);

	for (int i = 0; i < N; ++i) {
		std::cout << h_C[i] << ", ";
	}
	std::cout << std::endl;

	cudaFree(d_A);
	cudaFree(d_C);

	free(h_A);
	free(h_C);

	cudaDeviceReset();

	return 0;
}
