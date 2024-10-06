#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max Grid Size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;

    return 0;
}