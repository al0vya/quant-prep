#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// C = A @ B
// A: M x K
// B: K x N
// C: M x N <- M x K @ K x N
template<int BLOCK_SIDE_LENGTH>
__global__ void matmul_naive_kernel(float* A, float* B, float* C, int M, int K, int N) {
    const int bsl = BLOCK_SIDE_LENGTH;
    const int row = blockIdx.y * bsl + threadIdx.x / bsl;
    const int col = blockIdx.x * bsl + threadIdx.x % bsl;
    
    if (row >= M || col >= N) {
        return;
    }
    
    float* Arow = A + row * K;
    float* Bcol = B + col;
    
    float tmp = 0.0f;
    for (int i = 0; i < K; i++) {
        tmp += Arow[i] * Bcol[i * N];
    }
    C[row * N + col] = tmp;
}

template<int BLOCK_SIDE_LENGTH>
__global__ void matmul_smem_kernel(float* A, float* B, float* C, int M, int K, int N) {
    const int bsl = BLOCK_SIDE_LENGTH;
    const int row_l = threadIdx.x / bsl;
    const int col_l = threadIdx.x % bsl;
    const int row = blockIdx.y * bsl + row_l;
    const int col = blockIdx.x * bsl + col_l;
    
    if (row >= M || col >= N) {
        return;
    }
    
    __shared__ float s_A[BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH];
    __shared__ float s_B[BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH];
    
    float* Arow = A + row * K;
    float* Bcol = B + col;
    float* s_Arow = s_A + row_l * bsl;
    float* s_Bcol = s_B + col_l;
    
    float tmp = 0.0f;
    for (int k = 0; k < K; k += bsl) {
        float a = (k + col_l < K) ? Arow[k + col_l] : 0.0f;
        float b = (k + row_l < K) ? Bcol[(k + col_l) * N] : 0.0f;
        s_A[row_l * bsl + col_l] = a;
        s_B[row_l * bsl + col_l] = b;
        __syncthreads();
        
        for (int i = 0; i < bsl; i++) {
            tmp += s_Arow[i] * s_Bcol[i * bsl];
        }
        __syncthreads();
    }
    C[row * N + col] = tmp;
}

template<int BLOCK_SIDE_LENGTH>
__global__ void matmul_smem_1dtc_kernel(float* A, float* B, float* C, int M, int K, int N) {
    
}

int main() {
    const int M = 4000;
    const int N = 3500;
    const int K = 3000;
    const int BLOCK_SIDE_LENGTH = 32;
    const int bsl = BLOCK_SIDE_LENGTH;
    dim3 grid_size(CEIL_DIV(M, bsl), CEIL_DIV(N, bsl)); // grid size to launch at least M x N threads
    dim3 block_size(bsl * bsl); // 32 x 32 block flattened into 256 threads
    //matmul_kernel<BLOCK_SIDE_LENGTH><<<grid_size, block_size>>>(A, B, C, M, N, K);
}