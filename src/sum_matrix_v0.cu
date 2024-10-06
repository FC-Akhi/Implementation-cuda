// Summing Matrices with a 2D Grid and 2D Blocks


#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>


void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i=0; i<N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n",hostRef[i],gpuRef[i],i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}



void sumMatrixOnHost (float *A, float *B, float *C, const int nx, const int ny) {

    for (int iy = 0; iy < ny; iy++)

        for (int ix = 0; ix < nx; ix++)
        {
            C[(iy * nx) + ix] = A[(iy * nx) + ix] + B[(iy * nx) + ix];
        }

}


__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny) {
    
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*nx + ix;
    
    printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
        "gridDim:(%d, %d, %d) idx: %d\n", threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
        gridDim.x, gridDim.y, gridDim.z, idx);



    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];

}




void initialData(float *ip,int size) {
    
    
    for (int i=0; i<size; i++) {
    
        ip[i] = (float)( rand() & 0xFF )/10.0f;
    
    }

}



int main(int argc, char **argv) {

    printf("%s Starting...\n", argv[0]);

    // set up date size of matrix
    int nx = 8;
    int ny = 8;

    int nxy = nx*ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n",nx, ny);
    
    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);
    
    // initialize data at host side
    initialData (h_A, nxy);
    initialData (h_B, nxy);


    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    sumMatrixOnHost (h_A, h_B, hostRef, nx,ny);

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);



    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);


    // invoke kernel at host side
    int dimx = 4;
    int dimy = 4;

    dim3 block(dimx, dimy);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
    
    sumMatrixOnGPU2D <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
    
    
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
    
    
    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    
    
    // check device results
    checkResult(hostRef, gpuRef, nxy);
    
    
    // free device global memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);
    
    
    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    
    
    // reset device
    cudaDeviceReset();
    
    
    
    return (0);



}