#include<cuda_runtime.h>
#include<iostream>

#define CEIL(a, b) (((a) + (b) - 1) / (b))


__global__ void vectorAdd(float*c, float*a, float*b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}



int main(int argc, char const *argv[])
{
    int N = 1024;
    int n_bytes = N * sizeof(float);
    float* a = (float*)malloc(n_bytes);
    float* b = (float*) malloc(n_bytes);
    float* c = (float*) malloc(n_bytes);

    for (size_t i = 0; i < N; i++)
    {
        a[i] = 1;
        b[i] = 2;
    }

    float* ad;
    float* bd; 
    float* cd;

    cudaMalloc(&ad, n_bytes);
    cudaMalloc(&bd, n_bytes);
    cudaMalloc(&cd, n_bytes);

    cudaMemcpy(ad, a, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, n_bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size  = CEIL(N, block_size);
    vectorAdd<<<grid_size, block_size>>>(cd, ad, bd, N);

    cudaMemcpy(c, cd, n_bytes, cudaMemcpyDeviceToHost);

    std::cout << c[0] << ", " << c[1] << ", ... " << c[N-1] <<  std::endl;

    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);

    free(a), free(b), free(c);
    
    return 0;
}
