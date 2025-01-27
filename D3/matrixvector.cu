#include <cmath>
#include <cstdio>

__global__ void matrixVectorMul(const float* mat, const float* vec, float* out, int N){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N ){
        float sum = 0.0f;
        for (int col=0; col<N; col++){
            int idx = row * N + col;
            sum += mat[idx]*vec[col];
        }
        out[row] = sum;
    }

}

void main(){
    int N = 1000;
    const int SIZE = N*N*sizeof(float);

    float* mat = new float[N*N];
    float* vec = new float[N];
    float* out = new float[N];

    for (int i=0; i<N; i++){
        vec[i] = 1.0f;
        for (int j=0; j<N; j++){
            mat[i * N + j] = 1.0f;
        }
    }

    float* gmat, *gvec, *gout;
    cudaMalloc(&gmat, SIZE);
    cudaMalloc(&gvec, N*sizeof(float));
    cudaMalloc(&gout, N*sizeof(float));

    cudaMemcpy(gmat, mat, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(gvec, vec, N*sizeof(float), cudaMemcpyHostToDevice);


    //1D threads
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1)/threadsPerBlock;

    matrixVectorMul<<<numBlocks,threadsPerBlock>>>(gmat, gvec, gout, N);

    cudaMemcpy(out, gout, N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("out[%d] = %f\n", i, out[i]);
    }

    cudaFree(gmat);
    cudaFree(gvec);
    cudaFree(gout);

    delete[] mat;
    delete[] vec;
    delete[] out;

}