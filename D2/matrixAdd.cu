#include <cmath>

__global__ void matrixAddition(const float* A, const float* B, float* C, const int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N){
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    const int N = 1000;
    const int SIZE = N*N*sizeof(float);

    float* A = new float[N*N];
    float* B = new float[N*N];
    float* C = new float[N*N];

    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A,SIZE);
    cudaMalloc(&d_B,SIZE);
    cudaMalloc(&d_C,SIZE);

    cudaMemcpy(d_A, A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, B, SIZE, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixAddition<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, SIZE, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}