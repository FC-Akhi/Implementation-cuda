// This case works only for square matrix

#include <iostream>
#include <stdint.h>

unsigned int *generator(unsigned int img_width, unsigned int img_height, unsigned int tile_size, unsigned int *index_arr)
{
    
    unsigned int block_size = tile_size * tile_size;


    int index = 0;

    for (unsigned int i = 0; i < (img_width * img_height); i++){

        
        index = ((i % tile_size) + 
                (((i / tile_size) % (img_width  / tile_size)) * block_size) + 
                ((i / img_width) * tile_size) + 
                ((i / (img_width * tile_size))  * block_size));

        index_arr[i] = index;
        printf("i: %d, index: %d\n", i, index);

    }


    return index_arr;
}

void unit_testing(unsigned int img_width, unsigned int img_height, unsigned int tile_size, unsigned int *index_arr)
{

    unsigned int index_pal = 0;
    unsigned int block_size = tile_size * tile_size;

    
    for (unsigned int i = 0; i < (img_width * img_height); i++)
    {

        index_pal = (
            + ( (i/(img_width * tile_size)) * (img_width * tile_size) )
            + (i % tile_size)
            + ( ( (i / tile_size) % (img_width / tile_size) ) * block_size)
            + ( ((i/img_width) % tile_size) * tile_size)
        );

        
        printf("i: %d, index: %d\n", i, index_pal);
        
        
        if (index_pal != index_arr[i]){
            printf("Mismatched for %d\n", i);
            printf("i: %d, index_pal: %d\n", i, index_pal);
        }
        

    }
}



void initialData(float *ip,int size) {
    
    // generate different seed for random number
    time_t t;

    srand((unsigned) time(&t));
    
    for (int i=0; i<size; i++) {
    
        ip[i] = (float)( rand() & 0xFF )/10.0f;
    
    }

}

__global__ void offseting_GPU(float *A, const int nx, const int N) {

    // int i = threadIdx.x;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = iy * nx + ix;


    if (idx < N)
        


        printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) A[%d]\n", threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z, idx);


}


int main(void) {

    int nx = 4;
    int ny = 4;

    int nxy = nx*ny;
    int nBytes = nxy * sizeof(float);


    // malloc host global memory
    float *h_A;

    h_A = (float *)malloc (nBytes);
    

    // initialize data at host side
    initialData(h_A, nxy);


    // malloc device global memory
    float *d_A;
    cudaMalloc((float**)&d_A, nBytes);



    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);



    // invoke kernel at host side
    dim3 block (nx / 2, ny / 2);
    dim3 grid (block.x, block.y);

    printf("Execution configuration <<<%d, %d, %d, %d>>>\n", grid.x, grid.y, block.x, block.y);


    // double iStart = cpuSecond();
    offseting_GPU<<< grid, block >>>(d_A, nx, nxy);
    cudaDeviceSynchronize();
    // double iElaps = cpuSecond() - iStart;

    
    





    // // check device results
    // checkResult(hostRef, gpuRef, nElem);
    
    
    // free device global memory
    cudaFree(d_A);

    
    
    // free host memory
    free(h_A);

    
    return(0);

}