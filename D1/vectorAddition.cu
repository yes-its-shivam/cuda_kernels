#include <cmath>

__global__ void vectorAddition(const float* A, const float* B, float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    //initialisations
    const int N = 1000;
    const size_t arr_size = N*sizeof(float);
    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];
    float *d_a, *d_b,*d_c;
    
    // allocate memory on GPU
    cudaMalloc(&d_a, arr_size);
    cudaMalloc(&d_b, arr_size);
    cudaMalloc(&d_c, arr_size);

    //copy cpu vectors to gpu
    cudaMemcpy(d_a, A, arr_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, arr_size,cudaMemcpyHostToDevice);

    // threads, blocks and function call
    int blocksize=256;
    int gridsize = (N + blocksize - 1) / blocksize;
    vectorAddition<<<gridsize,blocksize>>>(d_a, d_b, d_c, N);

    //copy gpu vectors to cpu
    cudaMemcpy(C, d_c, arr_size, cudaMemcpyDeviceToHost);

    //freeup the memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] A;
    delete[] B;
    delete[] C;
}