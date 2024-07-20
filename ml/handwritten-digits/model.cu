// compile with: nvcc model.cu -O2 -o model.exe

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <iomanip> // for output file precision
#include <cuda_runtime.h>
#include <cooperative_groups.h> // CUDA cooperative groups
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

// Macros
#define CEIL_DIV(x, y) (x / y + ((x % y) != 0))
#define CHECK_CUDA_ERROR(ans) { cudaAssert((ans), __FILE__, __LINE__); }

// CUDA error handling
void cudaAssert(cudaError_t errorCode, const char* file, const int line) {
    if (errorCode != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(errorCode) << " " << file << " " << line << "\n";
        exit(errorCode);
    }
}

// Classes
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
                std::exit(EXIT_FAILURE);
            }
            
            CHECK_CUDA_ERROR(cudaMemcpy(m_data, data, size * sizeof(T), cudaMemcpyHostToDevice));
        }
        
        void copyToHostArray(T* data, size_t size) {
            if (m_size > size) {
                std::cerr << "Error: tried to copy to a host array that is smaller than the device array.\n";
                std::exit(EXIT_FAILURE);
            }
            
            CHECK_CUDA_ERROR(cudaMemcpy(data, m_data, m_size * sizeof(T), cudaMemcpyDeviceToHost));
        }
        
    private:
        int m_size;
        T* m_data;
};

// Utilities
bool are_floats_equal(float a, float b, float epsilon = 1e-6f) {
    return std::abs(a - b) <= epsilon;
}

void getMiniBatchFromHost(std::vector<float>& h_Xtrain, std::vector<int>& h_ytrain, CudaArray<float>& d_Xb, CudaArray<int>& d_yb,
                          int Xrows, int Xcols, int batchSize, int epoch) {
    int batchBegY = (batchSize * epoch) % Xrows; // row
    int batchBegX = batchBegY * Xcols;
    d_yb.copyFromHostArray(h_ytrain.data() + batchBegY, batchSize);
    d_Xb.copyFromHostArray(h_Xtrain.data() + batchBegX, batchSize * Xcols);
}

template <typename T>
T getMeanFromArray(T* d_in, size_t size) {
    std::vector<T> hSumOut(1);
    CudaArray<T>   dSumOut(1);
    
    // first call to get temporary storage
    void* tempStorage = nullptr;
    size_t tempStorageBytes = 0;
    CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, d_in, dSumOut.data(), size));
    CHECK_CUDA_ERROR(cudaMalloc(&tempStorage, tempStorageBytes));
    CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, d_in, dSumOut.data(), size));
    dSumOut.copyToHostArray(hSumOut.data(), hSumOut.size());
    
    return hSumOut.data()[0] / static_cast<T>(size);
}

template <typename T>
T readParam(const std::string& paramName, const std::string& filename = "params.txt") {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening " << filename << " for reading in hyperparameters.\n";
        std::exit(EXIT_FAILURE);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string name;
        T value;
        
        if (ss >> name >> value) {
            if (name == paramName) {
                return value;
            }
        }
    }
    
    std::cerr << "Error: hyperparameter " << paramName << " not found in " << filename << ".\n";
    std::exit(EXIT_FAILURE);
}

template <typename T>
std::vector<T> generateStandardNormalRandomVector(size_t N, int seed) {
    std::mt19937 g(seed); // generator
    std::normal_distribution<T> dis(0.0, 1.0);
    std::vector<T> randomVec(N);
    
    for (int i = 0; i < N; i++) {
        randomVec[i] = dis(g);
    }

    return randomVec;
}

// Kernels
__global__ void test_kernel(float* d_in, float* d_out, int nt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= nt) {
        return;
    }
    
    d_out[idx] = d_in[idx];
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

__global__ void matrixMultiplyAddKernel(float* d_out, float* d_inL, float* d_inR, float* d_b, int rowsL, int K, int colsR, bool bias = true) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / colsR;
    int col = idx % colsR;

    if (row < rowsL && col < colsR) {
        float value = (bias) ? d_b[col] : 0.0f;
        for (int i = 0; i < K; i++) {
            value += d_inL[row * K + i] * d_inR[i * colsR + col];
        }
        d_out[row * colsR + col] = value;
    }
}

__global__ void tanhKernel(float* d_out, float* d_in, int nt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= nt) {
        return;
    }
    
    d_out[idx] = tanhf(d_in[idx]);
}

__global__ void softmaxKernel(float* d_probs, float* d_logits, int B, int nHid2, bool backward = false) {
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
        p[i] = (!backward) ? max(1e-8f, expf(l[i] - maxx) / sum) : max(1e-8f, expf(l[i] - maxx) / sum) / B;
    }
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

__global__ void subtractOneDivBKernel(float* d_out, int* d_yb, int B, int nHid2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= B) {
        return;
    }
    
    float* out = d_out + idx * nHid2;
    
    int col = d_yb[idx];
    
    out[col] -= 1.0f / B;
}

__global__ void transposeKernel(float* d_out, const float* d_in, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= rows * cols) {
        return;
    }
    
    int row = idx / cols;
    int col = idx % cols;

    int idxT = col * rows + row;
    d_out[idxT] = d_in[idx];

}

__global__ void biasBackwardKernel(float* dbias, const float* dout, int rows, int cols) {
    extern __shared__ float smem[]; 
    const int warp_id = threadIdx.x / warpSize; 
    const int lane_id = threadIdx.x % warpSize; 
    const int tl = blockIdx.x * warpSize; 
    const int vstep = blockDim.x / warpSize; 
    const int col = tl + lane_id;
    
    if (col >= cols) { // handling cols % 32 != 0
        return;
    }
    
    const float* dout_col = dout + col;
    
    float dout_sum = 0.0f;
    for (int row = warp_id; row < rows; row += vstep) {
        dout_sum += (float)dout_col[row * cols];
    }
    smem[lane_id + warp_id * warpSize] = dout_sum;
    __syncthreads();
    
    dout_sum = 0.0f;
    if (warp_id == 0) {
        for (int j = 0; j < vstep; j++) {
            dout_sum += smem[lane_id + j * warpSize];
        }
        dbias[tl + lane_id] = (float)dbias[tl + lane_id] + dout_sum;
    }
}

__global__ void tanhBackwardKernel(float* d_dlin, float* d_dact, float* d_act, int nt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= nt) {
        return;
    }
    
    d_dlin[idx] = d_dact[idx] * (1 - d_act[idx] * d_act[idx]);
}

__global__ void parameterUpdateKernel(float* d_params, float* d_grads, float lr, int nt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= nt) {
        return;
    }
    
    d_params[idx] -= lr * d_grads[idx];
}

// File I/O
void readTrainingData(std::vector<float>& h_Xtrain, std::vector<int>& h_ytrain, const std::string& filename = "mnist_train.csv") {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error reading training data from " << filename << ".\n";
        std::exit(EXIT_FAILURE);
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

template <typename T>
void writeDeviceMatrixToFile(CudaArray<T>& d_out, int rows, int cols, const std::string& filename) {
    if (rows * cols != d_out.size()) {
        std::cerr << "Error: the number of entries to be written from the device matrix to " << filename << " does not match the matrix size.\n";
        std::exit(EXIT_FAILURE);
    }
    
    std::vector<T> h_out(d_out.size());
    d_out.copyToHostArray(h_out.data(), h_out.size());
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: failed to open file " << filename << " for writing device matrix to file.\n";
        std::exit(EXIT_FAILURE);
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
    if (rows != d_out.size()) {
        std::cerr << "Error: the number of entries to be written from the device vector to " << filename << " does not match the vector size.\n";
        std::exit(EXIT_FAILURE);
    }
    
    std::vector<T> h_out(d_out.size());
    d_out.copyToHostArray(h_out.data(), h_out.size());
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: failed to open file " << filename << " for writing device vector to file.\n";
        std::exit(EXIT_FAILURE);
    }
    
    file << std::fixed << std::setprecision(10);
    
    for (int j = 0; j < rows; j++) {
        file << h_out[j] << "\n";
    }
    
    file.close();
}

template <typename T>
void writeHostVectorToFile(std::vector<T> h_out, int rows, const std::string& filename) {
    if (rows != h_out.size()) {
        std::cerr << "Error: the number of entries to be written from the host vector to " << filename << " does not match the vector size.\n";
        std::exit(EXIT_FAILURE);
    }
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: failed to open file " << filename << " for writing host vector to file.\n";
        std::exit(EXIT_FAILURE);
    }
    
    file << std::fixed << std::setprecision(10);
    
    for (int j = 0; j < rows; j++) {
        file << h_out[j] << "\n";
    }
    
    file.close();
}

// Tests
void test(const std::string testMessage, bool test_cond) {
    if (test_cond) {
        std::cout << "Succeeded in " << testMessage << ".\n";
    } else {
        std::cout << "Failed in " << testMessage << ".\n";
        std::exit(EXIT_FAILURE);
    }
}

bool testCudaArray() {
    const int NUM_ELEMS = 1 << 20; // 2^20 
    std::vector<float> h_in(NUM_ELEMS);
    std::vector<float> h_out(NUM_ELEMS, 0.0f);
    
    for (int i = 0; i < NUM_ELEMS; i++) {
        h_in[i] = static_cast<float>(i);
    }
    
    CudaArray<float> d_in(h_in.data(), h_in.size());
    CudaArray<float> d_out(h_out.data(), h_out.size());
    
    const int blockSize = 256;
    const int numBlocks = NUM_ELEMS / blockSize; // powers of 2 always divisible
    const int numThreads = NUM_ELEMS;
    
    test_kernel<<<numBlocks, blockSize>>>(d_in.data(), d_out.data(), numThreads);
    
    d_out.copyToHostArray(h_out.data(), h_out.size());
    
    for (int i = 0; i < h_in.size(); i++) {
        if (!are_floats_equal(h_in[i], h_out[i])) {
            return false;
        }
    }
    
    return true;
}

int main(int argc, char** argv) {
    test("correctly initialising a CudaArray and then populating it with a kernel", testCudaArray());
    
    // tensor sizes, hyperparameters, CUDA parameters
    const int Xrows = 60000;
    const int Xcols = 784;
    const int batchSize = readParam<float>("batchSize");
    const int B = batchSize;
    const int nHid1 = 100;
    const int nHid2 = 10;
    bool testing = readParam<bool>("testing");
    const float lr = readParam<float>("lr");
    int epochs = readParam<int>("epochs");
    std::vector<float> losses(epochs);
    const int blockSize = 256;
    
    std::vector<float> h_Xtrain;
    std::vector<int>   h_ytrain;
    
    readTrainingData(h_Xtrain, h_ytrain);
    
    test("reading in the correct number of X training data (60000 rows x 784 columns)", h_Xtrain.size() == Xrows * Xcols);
    test("reading in the correct number of y training data (60000 rows)", h_ytrain.size() == Xrows);
    
    // parameter tensors
    CudaArray<float> d_W1(Xcols * nHid1);
    CudaArray<float> d_b1(nHid1);
    CudaArray<float> d_W2(nHid1 * nHid2);
    CudaArray<float> d_b2(nHid2);
    
    // layer tensors
    CudaArray<float> d_Xb(B * Xcols);
    CudaArray<int>   d_yb(B);
    CudaArray<float> d_lin(B * nHid1);
    CudaArray<float> d_act(B * nHid1);
    CudaArray<float> d_logits(B * nHid2);
    CudaArray<float> d_probs(B * nHid2);
    CudaArray<float> d_losses(B);
    
    // gradient tensors
    CudaArray<float> d_dlogits(B * nHid2);
    CudaArray<float> d_db2(nHid2);
    CudaArray<float> d_dW2(nHid1 * nHid2);
    CudaArray<float> d_dact(B * nHid1);
    CudaArray<float> d_dlin(B * nHid1);
    CudaArray<float> d_db1(nHid1);
    CudaArray<float> d_dW1(Xcols * nHid1);
    
    // transpose tensors
    CudaArray<float> d_XbT(Xcols * B);
    CudaArray<float> d_W2T(nHid2 * nHid1);
    CudaArray<float> d_actT(nHid1 * B);
    
    // populating tensors
    int seed = 21431;
    
    // weight tensors are initialised as random normal
    std::vector<float> rVec1 = generateStandardNormalRandomVector<float>(d_W1.size(), seed);
    std::vector<float> rVec2 = generateStandardNormalRandomVector<float>(d_W2.size(), seed);
    d_W1.copyFromHostArray(rVec1.data(), rVec1.size());
    d_W2.copyFromHostArray(rVec2.data(), rVec2.size());
    
    // biases, layer and gradient tensors can just be zero initialised
    cudaMemset(d_Xb.data(),      0, d_Xb.size());
    cudaMemset(d_yb.data(),      0, d_yb.size());
    cudaMemset(d_b1.data(),      0, d_b1.size());
    cudaMemset(d_b2.data(),      0, d_b2.size());
    cudaMemset(d_lin.data(),     0, d_lin.size());
    cudaMemset(d_act.data(),     0, d_act.size());
    cudaMemset(d_logits.data(),  0, d_logits.size());
    cudaMemset(d_probs.data(),   0, d_probs.size());
    cudaMemset(d_dlogits.data(), 0, d_dlogits.size());
    cudaMemset(d_db2.data(),     0, d_db2.size());
    cudaMemset(d_dW2.data(),     0, d_dW2.size());
    cudaMemset(d_dact.data(),    0, d_dact.size());
    cudaMemset(d_dlin.data(),    0, d_dlin.size());
    cudaMemset(d_db1.data(),     0, d_db1.size());
    cudaMemset(d_dW1.data(),     0, d_dW1.size());
    cudaMemset(d_XbT.data(),     0, d_XbT.size());
    cudaMemset(d_W2T.data(),     0, d_W2T.size());
    cudaMemset(d_actT.data(),    0, d_actT.size());
    
    for (int e = 0; e < epochs; e++) {
        // forward
        getMiniBatchFromHost(h_Xtrain, h_ytrain, d_Xb, d_yb, Xrows, Xcols, batchSize, e);
        matrixMultiplyAddKernel<<<CEIL_DIV(d_lin.size(), blockSize), blockSize>>>(d_lin.data(), d_Xb.data(), d_W1.data(), d_b1.data(), B, Xcols, nHid1);
        tanhKernel<<<CEIL_DIV(d_act.size(), blockSize), blockSize>>>(d_act.data(), d_lin.data(), d_act.size());
        matrixMultiplyAddKernel<<<CEIL_DIV(d_logits.size(), blockSize), blockSize>>>(d_logits.data(), d_act.data(), d_W2.data(), d_b2.data(), B, nHid1, nHid2);
        softmaxKernel<<<CEIL_DIV(B, blockSize), blockSize>>>(d_probs.data(), d_logits.data(), B, nHid2);
        crossEntropyKernel<<<CEIL_DIV(B, blockSize), blockSize>>>(d_losses.data(), d_probs.data(), d_yb.data(), B, nHid2);
        float loss = getMeanFromArray<float>(d_losses.data(), d_losses.size());
        losses[e] = loss;
        
        // backward
        softmaxKernel<<<CEIL_DIV(B, blockSize), blockSize>>>(d_dlogits.data(), d_logits.data(), B, nHid2, true);
        subtractOneDivBKernel<<<CEIL_DIV(B, blockSize), blockSize>>>(d_dlogits.data(), d_yb.data(), B, nHid2);
        transposeKernel<<<CEIL_DIV(d_W2.size(), blockSize), blockSize>>>(d_W2T.data(), d_W2.data(), nHid1, nHid2);
        matrixMultiplyAddKernel<<<CEIL_DIV(d_dact.size(), blockSize), blockSize>>>(d_dact.data(), d_dlogits.data(), d_W2T.data(), nullptr, B, nHid2, nHid1, false);
        transposeKernel<<<CEIL_DIV(d_actT.size(), blockSize), blockSize>>>(d_actT.data(), d_act.data(), B, nHid1);
        matrixMultiplyAddKernel<<<CEIL_DIV(d_dW2.size(), blockSize), blockSize>>>(d_dW2.data(), d_actT.data(), d_dlogits.data(), nullptr, nHid1, B, nHid2, false);
        biasBackwardKernel<<<CEIL_DIV(B, 32), blockSize, blockSize * sizeof(float)>>>(d_db2.data(), d_dlogits.data(), B, nHid2);
        tanhBackwardKernel<<<CEIL_DIV(d_dlin.size(), blockSize), blockSize>>>(d_dlin.data(), d_dact.data(), d_act.data(), d_dlin.size());
        transposeKernel<<<CEIL_DIV(d_Xb.size(), blockSize), blockSize>>>(d_XbT.data(), d_Xb.data(), B, Xcols);
        matrixMultiplyAddKernel<<<CEIL_DIV(d_dW1.size(), blockSize), blockSize>>>(d_dW1.data(), d_XbT.data(), d_dlin.data(), nullptr, Xcols, B, nHid1, false);
        biasBackwardKernel<<<CEIL_DIV(B, 32), blockSize, blockSize * sizeof(float)>>>(d_db1.data(), d_dlin.data(), B, nHid1);
        CHECK_CUDA_ERROR(cudaPeekAtLastError());
        
        // only save data for units tests in the first training epoch
        if (testing && e == 0) { 
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
            writeDeviceMatrixToFile<float>(d_dlogits, B, nHid2, "dlogits.csv");
            writeDeviceMatrixToFile<float>(d_dact, B, nHid1, "dact.csv");
            writeDeviceMatrixToFile<float>(d_W2T, nHid2, nHid1, "W2T.csv");
            writeDeviceMatrixToFile<float>(d_dW2, nHid1, nHid2, "dW2.csv");
            writeDeviceVectorToFile<float>(d_db2, nHid2, "db2.csv");
            writeDeviceMatrixToFile<float>(d_dlin, B, nHid1, "dlin.csv");
            writeDeviceMatrixToFile<float>(d_XbT, Xcols, B, "XbT.csv");
            writeDeviceMatrixToFile<float>(d_dW1, Xcols, nHid1, "dW1.csv");
            writeDeviceVectorToFile<float>(d_db1, nHid1, "db1.csv");
        }
        
        // parameter update
        parameterUpdateKernel<<<CEIL_DIV(d_W1.size(), blockSize), blockSize>>>(d_W1.data(), d_dW1.data(), lr, d_W1.size());
        parameterUpdateKernel<<<CEIL_DIV(d_b1.size(), blockSize), blockSize>>>(d_b1.data(), d_db1.data(), lr, d_b1.size());
        parameterUpdateKernel<<<CEIL_DIV(d_W2.size(), blockSize), blockSize>>>(d_W2.data(), d_dW2.data(), lr, d_W2.size());
        parameterUpdateKernel<<<CEIL_DIV(d_b2.size(), blockSize), blockSize>>>(d_b2.data(), d_db2.data(), lr, d_b2.size());
    }
    
    std::cout << "Finished training.\n";
    
    writeHostVectorToFile(losses, losses.size(), "training-losses.csv");
    
}