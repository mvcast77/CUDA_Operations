#include <iostream>

void checkGPU(){
	int dev_check = 0;
	cudaGetDeviceCount(&dev_check);

	if (dev_check) { std::cout << "No GPU available\n"; return; }

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

int main(int argc, char ** argv) {
	checkGPU();

	return 0;
}
