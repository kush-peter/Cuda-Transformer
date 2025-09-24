#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

typedef vector<vector<float>> Matrix;

// Generate a random matrix of size rows x cols
Matrix randomMatrix(int rows, int cols, float min_val = 0.0f, float max_val = 1.0f) {
    Matrix mat(rows, vector<float>(cols));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(min_val, max_val);

    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            mat[i][j] = dis(gen);
        }
    }
    return mat;
}



// CPU Matrix Multiplication 
Matrix matmulCPU(const Matrix& A, const Matrix& B){
    int M = A.size();
    int N = A[0].size();
    int K = B[0].size();

    if(B.size() != N){
        cerr << "Matrix dimensions do not match!\n";
        exit(EXIT_FAILURE);
    }

    Matrix C(M, vector<float>(K, 0));

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < K; ++j){
            float sum = 0;
            for(int k = 0; k < N; ++k){
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}


// GPU Matrix Multiplication
__global__ void matmulKernel(const float* A, const float* B, float* C, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < K){
        float val = 0;
        for(int i = 0; i < N; ++i){
            val += A[row*N+i] * B[i*K+col];
        }
        C[row*K+col] = val;
    }
}
Matrix matmulGPU(const Matrix& A, const Matrix& B){
    int M = A.size();
    int N = A[0].size();
    int K = B[0].size();

    if(B.size() != N){
        cerr << "Cant multiply!\n";
        exit(EXIT_FAILURE);
    }
    vector<float> A_flat(M*N);
    vector<float> B_flat(N*K);
    vector<float> C_flat(M*K,0);

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            A_flat[i*N+j] = A[i][j];
        }
    }
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < K; ++j){
            B_flat[i*K+j] = B[i][j];
        }
    }

    float *d_A, *d_B, *d_C;
    size_t size = sizeof(float);

    cudaMalloc(&d_A, M*N*size);
    cudaMalloc(&d_B, N*K*size);
    cudaMalloc(&d_C, M*K*size);

    cudaMemcpy(d_A, A_flat.data(), M*N*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat.data(), N*K*size, cudaMemcpyHostToDevice);

    //LAUNCH KERNEL 
    dim3 threads(16, 16);
    dim3 blocks((K+15) / 16, (M+15) / 16);

    matmulKernel<<<blocks, threads>>>(d_A,d_B,d_C,M,N,K);
    cudaDeviceSynchronize();

    cudaMemcpy(C_flat.data(), d_C, M*K*size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    Matrix C(M, vector<float>(K));
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < K; ++j){
            C[i][j] = C_flat[i*K+j];
        }
    }

    return C;
}

int main() {
    int size = 1024;
    Matrix A = randomMatrix(size, size);
    Matrix B = randomMatrix(size, size);

    cout << "Generated random 1024x1024 matrices A and B.\n";

    auto start = chrono::high_resolution_clock::now();
    Matrix C_cpu = matmulCPU(A, B);
    auto stop = chrono::high_resolution_clock::now();
    double cpu_ms = chrono::duration<double, milli>(stop-start).count();
    cout << "CPU MatMul Time: " 
        << cpu_ms << " ms\n";

    start = chrono::high_resolution_clock::now();
    Matrix C_gpu = matmulGPU(A, B);
    stop = chrono::high_resolution_clock::now();
    double gpu_ms = chrono::duration<double, milli>(stop-start).count();
    cout << "GPU MatMul Time: "
        << gpu_ms << " ms\n";

    cout << "Speedup: " << cpu_ms/gpu_ms << "x\n";

    cout << "C_cpu[0][0]: " << C_cpu[0][0] << ", C_gpu[0][0]: " << C_gpu[0][0] << endl;

}