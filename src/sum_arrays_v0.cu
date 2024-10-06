#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

double cpuSecond() {

    struct timeval tp;
    
    gettimeofday(&tp,NULL);
    
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);

}


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



void initialData(float *ip,int size) {
    
    // generate different seed for random number
    time_t t;

    srand((unsigned) time(&t));
    
    for (int i=0; i<size; i++) {
    
        ip[i] = (float)( rand() & 0xFF )/10.0f;
    
    }

}



void sumArrayOnHost(float *A, float *B, float *C, const int N) {

    for (int i = 0; i < N; i++)

        C[i] = A[i] + B[i];

}



__global__ void sumArrayOnGPU(float *A, float *B, float *C, const int N) {

    // int i = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
    //     "gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
    //     blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
    //     gridDim.x,gridDim.y,gridDim.z);

    if (i < N) C[i] = A[i] + B[i];

}


int main(void) {


    int nElem = 1<<24;

    printf("Vector size %d\n", nElem);


    size_t nBytes = nElem * sizeof(float);


    // malloc host global memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc (nBytes);
    h_B = (float *)malloc (nBytes);
    hostRef = (float *)malloc (nBytes);
    gpuRef = (float *)malloc (nBytes);


    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);


    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);


    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);


    // invoke kernel at host side
    int iLen = 1024;
    dim3 block (iLen);
    dim3 grid ((nElem + block.x - 1) / block.x);


    // dim3 block (1);
    // dim3 grid (nElem);

    double iStart = cpuSecond();
    sumArrayOnGPU<<< grid, block >>>(d_A, d_B, d_C, nElem);
    cudaDeviceSynchronize();
    double iElaps = cpuSecond() - iStart;

    printf("GPU execution time: %f seconds\n", iElaps);
    printf("Execution configuration <<<%d, %d>>>\n", grid.x,block.x);
    


    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    


    // add vector at host side for result checks
    iStart = cpuSecond();
    sumArrayOnHost(h_A, h_B, hostRef, nElem);
    iElaps = cpuSecond() - iStart;

    printf("CPU execution time: %f seconds\n", iElaps);



    // check device results
    checkResult(hostRef, gpuRef, nElem);
    
    
    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    
    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    
    
    return(0);

}