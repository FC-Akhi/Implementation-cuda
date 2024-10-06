/*
 * Description:
 * This program demonstrates the usage of CUDA grid and block dimensions in a simple kernel.
 * The kernel prints out the indices of threads, blocks, block dimensions, and grid dimensions.
 * It provides insight into how threads and blocks are structured within a grid.
 *
 * Date: 28-Sept, 2024
 * Reference: PROFESSIONAL CUDA ® C Programming by John Cheng; Page 33 LISTING 2-2: Check grid and block indices and dimensions (checkDimension.cu)
 */


#include <cuda_runtime.h>
#include <stdio.h>


__global__ void checkIndex(void) {

    printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
        "gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
        gridDim.x,gridDim.y,gridDim.z);

}



int main(int argc, char **argv) {

    // define total data element
    int nElem = 6;
    
    // define grid and block structure
    dim3 block (3);
    
    dim3 grid ((nElem+block.x-1)/block.x);
    
    // check grid and block dimension from host side
    printf("grid.x %d grid.y %d grid.z %d\n",grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n",block.x, block.y, block.z);
    
    // check grid and block dimension from device side
    checkIndex <<<grid, block>>> ();
    
    // reset device before you leave
    cudaDeviceReset();
    
    return(0);

}