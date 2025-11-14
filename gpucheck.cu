#include <iostream>

void checkGPU(){
	int dev_check = 0;
	cudaGetDeviceCount(&dev_check);

	if (!dev_check) { std::cout << "No GPU available\n"; return; }
	else if (dev_check > 1) std::cout << "More than one GPU is available\n\n";

	cudaDeviceProp info;
	int deviceID = 0;
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&info, deviceID);
	
	std::cout << "Name: " << info.name << std::endl \
	<< "Major/minor: " << info.major << "." << info.minor << std::endl \
	<< "Warp Size: " << info.warpSize << std::endl \
	<< "SM number: " << info.multiProcessorCount << std::endl \
	<< "Max threads per SM: " << info.maxThreadsPerMultiProcessor << std::endl \
	<< "Max threads per block: " << info.maxThreadsPerBlock << std::endl \
	<< "Memory Specifications" << std::endl \
	<< "Total shared memory for multiprocesser: " << info.sharedMemPerBlock << " bytes" << std::endl;
}

int main(int argc, char ** argv) {
	checkGPU();

	return 0;
}
