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
/*
[ t t t t | t t t t ]
|                   |
|                   |
|                   |
| t t t t | t t t t |
[                   ]
*/
template<int ITEM_BLOCK_WIDTH, int ITEM_BLOCK_HEIGHT, int ITEMS_PER_THREAD, int TILE_WIDTH_A, int TILE_HEIGHT_A>
__global__ void matmul_smem_1dtc_kernel(float* A, float* B, float* C, int M, int K, int N) {
    const int row_b = blockIdx.y * ITEM_BLOCK_HEIGHT; // row of item block
    const int row_l = threadIdx.x / ITEM_BLOCK_WIDTH; // local thread row in item block; used for stepping since each thread row processes multiple items
    const int row_t = threadIdx.x / TILE_WIDTH_A; // row in tile
    const int row_g = row_b + ITEMS_PER_THREAD * row_l + row_t;
    
    const int col_b = blockIdx.x * ITEM_BLOCK_WIDTH; // col of item block
    const int col_t = threadIdx.x % ITEM_BLOCK_WIDTH; // col in tile
    const int col_g = col_b + col_t;
    
    if (row_g >= M || col_g >= N) {
        return;
    }
    
    // tile A transposed
    const int TILE_HEIGHT_B = TILE_WIDTH_A;
    const int TILE_WIDTH_B = TILE_HEIGHT_A;
    
    __shared__ float s_A[TILE_WIDTH_A * TILE_HEIGHT_A];
    __shared__ float s_B[TILE_WIDTH_B * TILE_HEIGHT_B];
    
    float* Arow = A + row_g;
    float* Bcol = B + col_g;
    float* s_Arow = s_A + row_t * TILE_WIDTH_A;
    float* s_Bcol = s_B + col_t;
    float* s_Arow_i = s_A + row_l * ITEMS_PER_THREAD;
    float* s_Bcol_i = s_B + row_l;
            
    float tmp[ITEMS_PER_THREAD] = {0.0f};
    for (int k = 0; k < K; k += TILE_WIDTH_A) {
        float a = Arow[k + col_t];
        float b = Bcol[(k + col_t) * N];
        s_Arow[col_t] = a;
        s_Bcol[row_t * TILE_WIDTH_B] = b;
        __syncthreads();
        
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            for (int j = 0; j < TILE_WIDTH_A; j++) {
                tmp[i] += s_Arow_i[i * TILE_WIDTH_A + j] * s_Bcol[j * TILE_WIDTH_B + i];
            }
        }
        __syncthreads();
    }
    
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        C[(row_b + ITEMS_PER_THREAD * row_l + i) * N + col_t] = tmp[i];
    }
}

void matmul_naive() {
    const int M = 4000;
    const int N = 3500;
    const int K = 3000;
    const int BLOCK_SIDE_LENGTH = 32;
    const int bsl = BLOCK_SIDE_LENGTH;
    dim3 grid_dim(CEIL_DIV(M, bsl), CEIL_DIV(N, bsl)); // grid size to launch at least M x N threads
    dim3 block_dim(bsl * bsl); // 32 x 32 block flattened into 256 threads
    //matmul_naive_kernel<BLOCK_SIDE_LENGTH><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

void matmul_smem() {
    const int M = 4000;
    const int N = 3500;
    const int K = 3000;
    const int BLOCK_SIDE_LENGTH = 32;
    const int bsl = BLOCK_SIDE_LENGTH;
    dim3 grid_dim(CEIL_DIV(M, bsl), CEIL_DIV(N, bsl)); // grid size to launch at least M x N threads
    dim3 block_dim(bsl * bsl); // 32 x 32 block flattened into 256 threads
    //matmul_smem_kernel<BLOCK_SIDE_LENGTH><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

void matmul_smem_1dtc() {
    const int M = 4000;
    const int N = 3500;
    const int K = 3000;
    const int ITEM_BLOCK_WIDTH = 64; // cols of C per thread block
    const int ITEMS_PER_THREAD = 8;
    const int BLOCK_SIZE = 512;
    const int ITEM_BLOCK_HEIGHT = BLOCK_SIZE / ITEM_BLOCK_WIDTH * ITEMS_PER_THREAD;
    const int TILE_HEIGHT_A = ITEM_BLOCK_HEIGHT;
    const int TILE_WIDTH_A = BLOCK_SIZE / TILE_HEIGHT_A;
    dim3 grid_dim(CEIL_DIV(N, ITEM_BLOCK_WIDTH), CEIL_DIV(M, ITEM_BLOCK_HEIGHT));
    dim3 block_dim(BLOCK_SIZE);
    //matmul_smem_1dtc_kernel<ITEM_BLOCK_WIDTH, ITEM_BLOCK_HEIGHT, ITEMS_PER_THREAD, TILE_WIDTH_A, TILE_HEIGHT_A>
    //  <<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

int main() {
    
}

// C = A @ B
// A : M x K
// B : K x N
// C : M x N
template<int BLOCK_SIDE_LENGTH>
__global__ matmul_naive(float* A, float* B, float* C, int M, int K, int N) {
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
__global__ matmul_smem(float* A, float* B, float* C, int M, int K, int N) {
    const int bsl = BLOCK_SIDE_LENGTH;
    const int row_t = threadIdx.x / bsl;
    const int col_t = threadIdx.x % bsl;
    const int row = blockIdx.y * bsl + row_t;
    const int col = blockIdx.x * bsl + col_t;
    
    if (row >= M || col >= N) {
        return;
    }
    
    __shared__ s_A[BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH];
    __shared__ s_b[BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH];
    
    float* Arow = A + row * K;
    float* Bcol = B + col;
    float* s_Arow = s_A + row_t * bsl;
    float* s_Bcol = s_B + col_t;
    
    float tmp = 0.0f;
    for (int k = 0; k < K; k += bsl) {
        float a = Arow[k + col_t];
        float b = Bcol[(k + col_t) * N];
        s_Arow[col_t] = a;
        s_Bcol[row_t * bsl] = b;
        __syncthreads();
        
        for (int i = 0; i < bsl; i++) {
            tmp += s_Arow[i] * s_Bcol[i * bsl];
        }
        __syncthreads();
    }
    C[row * N + col] = tmp;
}

template<int ITEM_BLOCK_WIDTH, int ITEM_BLOCK_HEIGHT, int ITEMS_PER_THREAD, int TILE_HEIGHT_A, int TILE_WIDTH_A>
__global__ void matmul_1dtc(float* A, float* B, float* C, int M, int K, int N) {
    const int row_b = blockIdx.y * ITEM_BLOCK_HEIGHT;
    const int row_l = threadIdx.x / ITEM_BLOCK_WIDTH;
    const int row_t = threadIdx.x / TILE_WIDTH_A;
    const int row_g = row_b + row_l * ITEMS_PER_THREAD + row_t;
    
    const int col_b = blockIdx.x + ITEM_BLOCK_WIDTH;
    const int col_t = threadIdx.x % TILE_WIDTH_A;
    const int col_g = col_b + col_g;
    
    if (row_g >= M || col_g >= N) {
        return;
    }
    
    const int TILE_HEIGHT_B = TILE_WIDTH_A;
    const int TILE_WIDTH_B = TILE_HEIGHT_A;
    
    __shared__ s_A[TILE_HEIGHT_A * TILE_WIDTH_A];
    __shared__ s_b[TILE_HEIGHT_B * TILE_WIDTH_B];
    
    float* Arow = A + row_g * K;
    float* Bcol = B + col_g;
    float* s_Arow = s_A + row_t * TILE_WIDTH_A;
    float* s_Bcol = s_B + col_t;
    float* s_Arow_t = s_A + row_l * ITEMS_PER_THREAD * TILE_WIDTH_A;
    float* s_Bcol_t = s_B + row_l * ITEMS_PER_THREAD;
    
    float tmp[ITEMS_PER_THREAD] = {0.0f};
    for (int k = 0; k < K; k += TILE_WIDTH_A) {
        float a = Arow[k + col_t];
        float b = Bcol[(k + row_t) * N];
        s_Arow[col_t] = a;
        s_Bcol[row_t * TILE_WIDTH_B] = b;
        __syncthreads();
        
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            for (int j = 0; j < TILE_WIDTH_A; j++) {
                tmp[i] = s_Arow_t[i * TILE_WIDTH_A] * s_Bcol_t[j * TILE_WIDTH_B + i];
            }
        }
        __syncthreads();
    }
    
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        C[(row_b + row_l * ITEMS_PER_THREAD + i) * N + col_g] = tmp[i];
    }
}

/*
C = 
[ t t t t t t t t t |   ]
|                   |   |
|                   |   |
|                   |   |
| t t t t t t t t t |   |
|                   |   |
|                   |   |
|                   |   |
| ------------------|-- |
[                   |   ]
*/
#define CEIL_DIV(A, B) ((A + B - 1) / B)
void thread_coarsening() {
    const int M = 4096;
    const int K = 4096;
    const int N = 4096;
    const int ITEM_BLOCK_WIDTH = 64;
    const int ITEMS_PER_THREAD = 8;
    const int BLOCK_SIZE = 512;
    const int ITEM_BLOCK_HEIGHT = BLOCK_SIZE / ITEM_BLOCK_WIDTH * ITEMS_PER_THREAD;
    dim3 grid_dim(CEIL_DIV(M, ITEM_BLOCK_HEIGHT), CEIL_DIV(N, ITEM_BLOCK_WIDTH));
    dim3 block_dim(BLOCK_SIZE);
    const int TILE_HEIGHT_A = ITEM_BLOCK_HEIGHT;
    const int TILE_WIDTH_A = BLOCK_SIZE / TILE_HEIGHT_A;
}