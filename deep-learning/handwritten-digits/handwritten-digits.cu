#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip> // for output file precision
#include <cuda_runtime.h>
#include <cooperative_groups.h> // CUDA cooperative groups
#include <cooperative_groups/reduce.h> // CUDA cooperative groups

#define CEIL_DIV(x, y) (x / y + ((x % y) != 0))

#define CHECK_CUDA_ERROR(ans) { cudaAssert((ans), __FILE__, __LINE__); }

void cudaAssert(cudaError_t errorCode, const char* file, const int line) {
    if (errorCode != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(errorCode) << " " << file << " " << line << "\n";
        exit(errorCode);
    }
}

template <typename T>
class CudaArray {
    public:
        CudaArray(size_t size) : m_size(size), m_data(nullptr) {
            CHECK_CUDA_ERROR(cudaMalloc(&m_data, size * sizeof(T)))
        }
        
        CudaArray(T* data, size_t size) : m_size(size), m_data(nullptr) {
            CHECK_CUDA_ERROR(cudaMalloc(&m_data, size * sizeof(T)))
            copyFromHostArray(data, size);
        }
        
        ~CudaArray() {
            if (m_data != nullptr) {
                CHECK_CUDA_ERROR(cudaFree(m_data))
            }
        }
        
        // deleting copy semantics and having move semantics instead seems to go hand in hand
        CudaArray(const CudaArray&) = delete;
        
        CudaArray& operator=(const CudaArray&) = delete;
        
        // making a completely new instance of this class
        CudaArray(const CudaArray&& other) : m_size(other.m_size), m_data(other.m_data) {
            other.m_data = nullptr;
        }
        
        // assigning to an existing instance of this class, which needs to free its resources
        // before acquiring other's resources
        CudaArray& operator=(CudaArray&& other) {
            // self assignment checking and freeing existing resources
            if (this != &other) {
                if (this->m_data != nullptr) {
                    CHECK_CUDA_ERROR(cudaFree(this->m_data));
                }
            }
            
            // acquiring other's resources
            this->m_size = other.m_size;
            this->m_data = other.m_data;
            
            // preventing other from freeing its resources, which now actually belong to this instance of the class
            other.m_size = 0;
            other.m_data = nullptr;
        }
        
        int size() {
            return m_size;
        }
        
        T* data() {
            return m_data;
        }
        
        void copyFromHostArray(T* data, size_t size) {
            if (m_size < size) {
                std::cerr << "Error: tried to copy from a host array that is larger than the device array.\n";
                exit(1);
            }
            
            CHECK_CUDA_ERROR(cudaMemcpy(m_data, data, size * sizeof(T), cudaMemcpyHostToDevice));
        }
        
        void copyToHostArray(T* data, size_t size) {
            if (m_size > size) {
                std::cerr << "Error: tried to copy to a host array that is smaller than the device array.\n";
                exit(1);
            }
            
            CHECK_CUDA_ERROR(cudaMemcpy(data, m_data, m_size * sizeof(T), cudaMemcpyDeviceToHost));
        }
        
    private:
        int m_size;
        T* m_data;
};

bool are_floats_equal(float a, float b, float epsilon = 1e-6f) {
    return std::abs(a - b) <= epsilon;
}

__global__ void test_kernel(float* d_in, float* d_out, int nt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= nt) {
        return;
    }
    
    d_out[idx] = d_in[idx];
}

bool testCudaArray() {
    constexpr int NUM_ELEMS = 1 << 20; // 2^20 
    std::vector<float> h_in(NUM_ELEMS);
    std::vector<float> h_out(NUM_ELEMS, 0.0f);
    
    for (int i = 0; i < NUM_ELEMS; i++) {
        h_in[i] = static_cast<float>(i);
    }
    
    CudaArray<float> d_in(h_in.data(), h_in.size());
    CudaArray<float> d_out(h_out.data(), h_out.size());
    
    constexpr int blockSize = 256;
    constexpr int numBlocks = NUM_ELEMS / blockSize; // powers of 2 always divisible
    constexpr int numThreads = NUM_ELEMS;
    
    test_kernel<<<numBlocks, blockSize>>>(d_in.data(), d_out.data(), numThreads);
    
    d_out.copyToHostArray(h_out.data(), h_out.size());
    
    for (int i = 0; i < h_in.size(); i++) {
        if (!are_floats_equal(h_in[i], h_out[i])) {
            return false;
        }
    }
    
    return true;
}

// linear congruential generator
__device__ float uniformRandomNumber(int seed, int idx) {
    // constants from Numerical Recipes
    const int a = 1664525;
    const int c = 1013904223;
    int state = seed + idx;
    state = a * state + c;
    state = a * state + c;
    return ((state & 0x7FFFFFFF) / static_cast<float>(0x80000000));
}

__global__ void populateTensorRandomNormalKernel(float* d_out, int nt, float mean, float stddev, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nt) {
        return;
    }
    
    float u1 = uniformRandomNumber(seed, idx);
    float u2 = uniformRandomNumber(seed, idx + nt);

    // Box-Muller transform
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * 3.141592f * u2;

    // generating normally distributed random number
    float z0 = r * cosf(theta);

    d_out[idx] = mean + z0 * stddev;
}

void readTrainingData(std::vector<float>& h_Xtrain, std::vector<int>& h_ytrain, const std::string& filename = "mnist_train.csv") {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error reading training data from " << filename << ".\n";
        exit(1);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string c;
        std::getline(ss, c, ','); // read first column; stop at , delimiter
        int y = std::stoi(c);
        h_ytrain.push_back(y);
        
        while (std::getline(ss, c, ',')) {
            float x = std::stof(c);
            h_Xtrain.push_back(x);
        }
    }
    
    file.close();
}

void test(const std::string testMessage, bool test_cond) {
    if (test_cond) {
        std::cout << "Succeeded in " << testMessage << ".\n";
    } else {
        std::cout << "Failed in " << testMessage << ".\n";
        exit(1);
    }
}

void getMiniBatchFromHost(std::vector<float>& h_Xtrain, std::vector<int>& h_ytrain, CudaArray<float>& d_Xb, CudaArray<int>& d_yb,
                          int Xrows, int Xcols, int batchSize, int epoch) {
    int batchBegX = (batchSize * Xcols * epoch) % (Xrows * Xcols);
    int batchBegY = (batchSize * epoch) % Xrows;
    d_Xb.copyFromHostArray(h_Xtrain.data() + batchBegX, batchSize * Xcols);
    d_yb.copyFromHostArray(h_ytrain.data() + batchBegY, batchSize);
}

template <typename T>
void writeDeviceMatrixToFile(CudaArray<T>& d_out, int rows, int cols, const std::string& filename) {
    std::vector<T> h_out(d_out.size());
    d_out.copyToHostArray(h_out.data(), h_out.size());
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: failed to open file " << filename << " for writing device matrix to file.\n";
        exit(-1);
    }
    
    file << std::fixed << std::setprecision(10);
    
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            file << h_out[j * cols + i];
            if (i < cols - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    
    file.close();
}

template <typename T>
void writeDeviceVectorToFile(CudaArray<T>& d_out, int rows, const std::string& filename) {
    std::vector<T> h_out(d_out.size());
    d_out.copyToHostArray(h_out.data(), h_out.size());
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: failed to open file " << filename << " for writing device vector to file.\n";
        exit(-1);
    }
    
    file << std::fixed << std::setprecision(10);
    
    for (int j = 0; j < rows; j++) {
        file << h_out[j] << "\n";
    }
    
    file.close();
}

// d_out = d_inL x d_inR + d_b
// d_inL: rowsL x K
// d_inR: K x colsR
// d_out: rowsL x colsR
// d_b: 1 x colsR
__global__ void matrixMultiplyAddKernel(float* d_out, float* d_inL, float* d_inR, float* d_b, int rowsL, int K, int colsR) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / colsR;
    int col = idx % colsR;

    if (row < rowsL && col < colsR) {
        float value = 0.0f;
        for (int i = 0; i < K; i++) {
            value += d_inL[row * K + i] * d_inR[i * colsR + col];
        }
        d_out[row * colsR + col] = value + d_b[col];
    }
}

__global__ void tanhKernel(float* d_out, float* d_in, int nt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= nt) {
        return;
    }
    
    d_out[idx] = tanhf(d_in[idx]);
}

__global__ void softmaxKernel(float* d_probs, float* d_logits, int B, int nHid2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= B) {
        return;
    }
    
    float* l = d_logits + idx * nHid2;
    
    float maxx = -FLT_MAX;
    for (int i = 0; i < nHid2; i++) {
        maxx = max(maxx, l[i]);
    }
    
    float sum = 0.0f;
    for (int i = 0; i < nHid2; i++) {
        sum += expf(l[i] - maxx);
    }
    
    float* p = d_probs + idx * nHid2;
    for (int i = 0; i < nHid2; i++) {
        p[i] = max(1e-8f, expf(l[i] - maxx) / sum);
    }
    
    /*namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // meta_group_size is the number of warps in a block, and meta_group_rank is the warp index
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= B) {
        return;
    }
    
    // the row of input that this group of threads is responsible for
    const float* l = d_logits + idx * nHid2;
    
    // max
    float maxx = -FLT_MAX;
    for (int i = warp.thread_rank(); i < nHid2; i += warp.size()) {
        maxx = max(maxx, l[i]);
    }
    maxx = cg::reduce(warp, maxx, cg::greater<float>{});
    
    // sum
    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < nHid2; i += warp.size()) {
        sum += expf(l[i] - maxx);
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    
    // final normalisation
    float* p = d_probs + idx * nHid2;
    for (int i = warp.thread_rank(); i < nHid2; i += warp.size()) {
        p[i] = expf(l[i] - maxx) / sum;
    }*/
}

__global__ void crossEntropyKernel(float* d_losses, float* d_probs, int* d_yb, int B, int nHid2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= B) {
        return;
    }
    
    float* p = d_probs + idx * nHid2;
    
    int col = d_yb[idx];
    
    d_losses[idx] = -logf(p[col]);
}

int main(int argc, char** argv) {
    test("correctly initialising a CudaArray and then populating it with a kernel", testCudaArray());
    
    // tensor sizes
    constexpr int Xrows = 60000;
    constexpr int Xcols = 784;
    constexpr int batchSize = 2000;
    constexpr int B = batchSize;
    constexpr int nHid1 = 100;
    constexpr int nHid2 = 10;
    
    std::vector<float> h_Xtrain;
    std::vector<int>   h_ytrain;
    
    readTrainingData(h_Xtrain, h_ytrain);
    
    test("reading in the correct number of X training data (60000 rows x 784 columns)", h_Xtrain.size() == Xrows * Xcols);
    test("reading in the correct number of y training data (60000 rows)", h_ytrain.size() == Xrows);
    
    // parameter and layer tensors
    CudaArray<float> d_Xb(B * Xcols);
    CudaArray<int>   d_yb(B);
    CudaArray<float> d_W1(Xcols * nHid1);
    CudaArray<float> d_b1(nHid1);
    CudaArray<float> d_lin(B * nHid1);
    CudaArray<float> d_act(B * nHid1);
    CudaArray<float> d_W2(nHid1 * nHid2);
    CudaArray<float> d_b2(nHid2);
    CudaArray<float> d_logits(B * nHid2);
    CudaArray<float> d_probs(B * nHid2);
    CudaArray<float> d_losses(B);
    
    // populating tensors
    int blockSize = 256;
    float mean = 0.0f;
    float stddev = 1.0f;
    int seed = 21431;
    
    // parameter tensors are initialised as random normal
    populateTensorRandomNormalKernel<<<CEIL_DIV(d_W1.size(), blockSize), blockSize>>>(d_W1.data(), d_W1.size(), mean, stddev, seed);
    populateTensorRandomNormalKernel<<<CEIL_DIV(d_b1.size(), blockSize), blockSize>>>(d_b1.data(), d_b1.size(), mean, stddev, seed);
    populateTensorRandomNormalKernel<<<CEIL_DIV(d_W2.size(), blockSize), blockSize>>>(d_W2.data(), d_W2.size(), mean, stddev, seed);
    populateTensorRandomNormalKernel<<<CEIL_DIV(d_b2.size(), blockSize), blockSize>>>(d_b2.data(), d_b2.size(), mean, stddev, seed);
    
    // layer tensors can just be zero initialised
    cudaMemset(d_Xb.data(),     0, d_Xb.size());
    cudaMemset(d_yb.data(),     0, d_yb.size());
    cudaMemset(d_lin.data(),    0, d_lin.size());
    cudaMemset(d_act.data(),    0, d_act.size());
    cudaMemset(d_logits.data(), 0, d_logits.size());
    cudaMemset(d_probs.data(),  0, d_probs.size());
    
    // forward
    getMiniBatchFromHost(h_Xtrain, h_ytrain, d_Xb, d_yb, Xrows, Xcols, batchSize, 1);
    matrixMultiplyAddKernel<<<CEIL_DIV(d_lin.size(), blockSize), blockSize>>>(d_lin.data(), d_Xb.data(), d_W1.data(), d_b1.data(), B, Xcols, nHid1);
    tanhKernel<<<CEIL_DIV(d_act.size(), blockSize), blockSize>>>(d_act.data(), d_lin.data(), d_act.size());
    matrixMultiplyAddKernel<<<CEIL_DIV(d_logits.size(), blockSize), blockSize>>>(d_logits.data(), d_act.data(), d_W2.data(), d_b2.data(), B, nHid1, nHid2);
    softmaxKernel<<<CEIL_DIV(B, blockSize), blockSize>>>(d_probs.data(), d_logits.data(), B, nHid2);
    crossEntropyKernel<<<CEIL_DIV(B, blockSize), blockSize>>>(d_losses.data(), d_probs.data(), d_yb.data(), B, nHid2);
    
    CHECK_CUDA_ERROR(cudaPeekAtLastError())
    
    // writing data for testing against pytorch
    writeDeviceMatrixToFile<float>(d_Xb, batchSize, Xcols, "Xb.csv");
    writeDeviceVectorToFile<int>  (d_yb, batchSize, "yb.csv");
    writeDeviceMatrixToFile<float>(d_W1, Xcols, nHid1, "W1.csv");
    writeDeviceVectorToFile<float>(d_b1, nHid1, "b1.csv");
    writeDeviceMatrixToFile<float>(d_lin, B, nHid1, "lin.csv");
    writeDeviceMatrixToFile<float>(d_act, B, nHid1, "act.csv");
    writeDeviceMatrixToFile<float>(d_W2, nHid1, nHid2, "W2.csv");
    writeDeviceVectorToFile<float>(d_b2, nHid2, "b2.csv");
    writeDeviceMatrixToFile<float>(d_logits, B, nHid2, "logits.csv");
    writeDeviceMatrixToFile<float>(d_probs, B, nHid2, "probs.csv");
    writeDeviceVectorToFile<float>(d_losses, B, "losses.csv");
    
}