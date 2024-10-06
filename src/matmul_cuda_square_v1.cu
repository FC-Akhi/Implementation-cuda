#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <sstream>  // For stringstream
#include <cstdlib>  // For exit()

// Matrix size (for simplicity, we assume a square matrix)



void cpu_matrix_mult_baseline(float *mat_a, float *mat_b, float *mat_res, int row_a, int col_a, int col_b)
{
    for (int i = 0; i < row_a; ++i) 
    {
        for (int j = 0; j < col_b; ++j) 
        {
            float tmp = 0.0f;
            for (int k = 0; k < col_a; ++k)
            {
                tmp += mat_a[(i * col_a) + k] * mat_b[(k * col_b) + j];
            }
            mat_res[(i * col_b) +j] = tmp;
        }
    }
}


void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


__global__ void tile_matrix_multiply(float *A, float *B, float *C, int tile_width, int width) {

    // Shared memory for storing tile portions of A and B
    __shared__ float shareA[2][2];
    __shared__ float shareB[2][2];

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate row and column index for matrix C
    int ix = bx * tile_width + tx; // col of mat
    int iy = by * tile_width + ty; // row of mat


    printf("Block (%d, %d), Thread (%d, %d): Computing C[%d][%d]\n", bx, by, tx, ty, iy, ix);

    // Initialize the sum accumulator for the final result
    float temp = 0.0f;

    // Loop over tiles of the input matrices A and B
    for(int i = 0; i < width/tile_width; ++i) {

        // Load a tile of matrix A and matrix B into shared memory
        printf("A: %.2f \n", A[(iy * width) + (i * tile_width + tx)]);
        printf("B: %.2f \n", B[((i * tile_width) + ty) * width + ix]);

        shareA[ty][tx] = A[(iy * width) + (i * tile_width + tx)];
        shareB[ty][tx] = B[((i * tile_width) + ty) * width + ix];

        // Synchronize to ensure all threads have loaded their data
        __syncthreads();

        // Perform the computation for the current tile
        for(int k = 0; k < tile_width; ++k) {

            temp += shareA[ty][k] * shareB[k][tx];
            __syncthreads();

        }

        // Synchronize again before loading the next tile
        __syncthreads();

    }

    // Write the result to the output matrix C
    C[iy * width + ix] = temp;


}

int main() {

    // Matrix dimensions
    int width;
    std::cout << "Enter size of matrix: " << std::endl;
    std::cin >> width;

    size_t size = width * width * sizeof(float);

    // Host pointers
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);



    // Get input for matrix A from the user
    std::cout << "Enter values for matrix A (row-major order): " << std::endl;
    for (int i = 0; i < width * width; ++i) {
        std::cin >> h_A[i];
    }

    // Get input for matrix B from the user
    std::cout << "Enter values for matrix B (row-major order): " << std::endl;
    for (int i = 0; i < width * width; ++i) {
        std::cin >> h_B[i];
    }

    // Initialize matrices A and B on the host
    for (int i = 0; i < width * width; ++i) {
        h_C[i] = 0.0f;  // Initialize result matrix to zero
    }



    // Device pointers
    float *d_A, *d_B, *d_C;

    // Allocate device memory for matrices A, B, and C
    checkCudaError(cudaMalloc((void **)&d_A, size), "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc((void **)&d_B, size), "Failed to allocate device memory for B");
    checkCudaError(cudaMalloc((void **)&d_C, size), "Failed to allocate device memory for C");

    // Copy matrices A and B from host to device
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Failed to copy A from host to device");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Failed to copy B from host to device");

    // Define block size and grid size (assuming each block processes 2x2 tiles)
    dim3 blockSize(2, 2);  // 2x2 threads per block (small, for simplicity)
    dim3 gridSize(width / 2, width / 2);  // Enough blocks to cover the entire matrix

    // Launch the kernel
    tile_matrix_multiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, 2, width);

    // Check for any errors during kernel execution
    checkCudaError(cudaGetLastError(), "Failed to launch kernel");
    checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize device after kernel launch");

    // Copy the result matrix C from device to host
    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "Failed to copy C from device to host");



    // Print some of the results (for verification)
    std::cout << "Results of C:" << std::endl;
    for (int i = 0; i < width * width; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;




    //============Verification=========================
    // Initialize temp_C with the correct size
    float temp_C[width * width];

    // Initialize the result matrix to zero
    for (int i = 0; i < width * width; ++i) {
        temp_C[i] = 0.0f;
    }


    // Baseline matmul is called
    cpu_matrix_mult_baseline(h_A, h_B, temp_C, width, width, width);

    // Print some of the results (for verification)
    std::cout << "Results of C_baseline:" << std::endl;
    for (int i = 0; i < width * width; ++i) {
        std::cout << temp_C[i] << " ";
    }
    std::cout << std::endl;


    const float tolerance = 1e-5;  // Allow a small tolerance for floating-point comparison
    bool mismatch = false;

    for (int i = 0; i < width * width; ++i) {
        if (fabs(h_C[i] - temp_C[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": GPU result = " << h_C[i] << ", CPU result = " << temp_C[i] << std::endl;
            mismatch = true;
        }
    }


    if (!mismatch) {
        std::cout << "Results match!" << std::endl;
    }




    // Free device memory
    checkCudaError(cudaFree(d_A), "Failed to free device memory for A");
    checkCudaError(cudaFree(d_B), "Failed to free device memory for B");
    checkCudaError(cudaFree(d_C), "Failed to free device memory for C");

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device (optional, good practice)
    checkCudaError(cudaDeviceReset(), "Failed to reset device");

    return 0;
}