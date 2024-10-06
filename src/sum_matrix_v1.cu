// Summing Matrices with a 1D Grid and 1D Blocks


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
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];  // Element-wise addition
        }
        ia += nx;  // Move to the next row in matrix A
        ib += nx;  // Move to the next row in matrix B
        ic += nx;  // Move to the next row in matrix C
    }
}





void initialData(float *ip,int size) {
    
    
    for (int i=0; i<size; i++) {
    
        ip[i] = (float)( rand() & 0xFF )/10.0f;
    
    }

}


__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC, int nx, int ny) {

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (ix < nx ) {

        for (int iy = 0; iy < ny; iy++) {

            int idx = iy*nx + ix;
        
            MatC[idx] = MatA[idx] + MatB[idx];
        
        }
    }

}



int main(int argc, char **argv) {

    printf("%s Starting...\n", argv[0]);

    // set up date size of matrix
    int nx = 64;
    int ny = 64;

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
    int dimx = 32;
    int dimy = 1;

    dim3 block(dimx, dimy);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
    
    sumMatrixOnGPU1D <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
    
    
    printf("sumMatrixOnGPU1D <<<(%d,%d), (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
    
    
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