#include <iostream>

void checkGPU(){
	cudaDeviceProp info;
	int deviceID = 0;
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&info, deviceID);
	
	std::cout << "Name: " << info.name << std::endl;
	std::cout << "Major/minor: " << info.major << "." << info.minor << std::endl;
	std::cout << "Warp Size: " << info.warpSize << std::endl;
	std::cout << "SM number: " << info.multiProcessorCount << std::endl;
	std::cout << "Max threads per SM: " << info.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "Max threads per block: " << info.maxThreadsPerBlock << std::endl;
	std::cout << "Memory Specifications" << std::endl;
	std::cout << "Total shared memory for multiprocesser: " << info.sharedMemPerBlock << " bytes" << std::endl;
}

// __device__ void gauss(float * A, float * C){
// 	int i = threadIdx.x;
// 	C[i] = C[i] + A[i];
// }

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


	cudaFree(d_A);
	cudaFree(d_C);

	free(h_A);
	free(h_C);

	return 0;
}
