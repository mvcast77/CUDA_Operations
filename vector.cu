#include <iostream>

void checkGPU(){
	cudaDeviceProp info;
	int deviceID = 0;
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&info, deviceID);
	
	std::cout << "Name: " << info.name << std::endl;
	std::cout << "Major/minor: " << info.major << "." << info.minor << std::endl;
	std::cout << "Warp Size: " << info.warpSize << std::endl;
}

__device__ void divide(float * C){
	int i = threadIdx.x;
	C[i] /= 2;
}

__global__ void gauss(float * A, float * C, int size){
	int i = threadIdx.x;
	C[i] = C[size - 1 - i] + A[i];
	// divide(C);
}

int main(int argc, char ** argv) {
	checkGPU();

	float * h_A, * d_A, * h_C, * d_C;

	h_A = (float*) malloc(10 *sizeof(float));
	h_C = (float*) malloc(10 *sizeof(float));

	cudaMalloc((float**) &d_A, 10 * sizeof(float));
	cudaMalloc((float**) &d_C, 10 * sizeof(float));

	for (int i = 0; i < 10; ++i){
		h_A[i] = i;
		h_C[i] = i;
	}

	cudaMemcpy(d_A, h_A, 10 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, 10 * sizeof(float), cudaMemcpyHostToDevice);

	int threadperblock = 10;
	int numBlocks = 1;

	gauss<<<numBlocks, threadperblock>>>(d_A, d_C, 10);

	cudaDeviceSynchronize();

	cudaMemcpy(h_A, d_A, 10 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_C, d_C, 10 * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	for (int i = 0; i < 10; ++ i){
		std::cout << h_C[i] << ", ";
	}
	std::cout << std::endl;

	cudaFree(d_A);
	cudaFree(d_C);

	free(h_A);
	free(h_C);

	return 0;
}
