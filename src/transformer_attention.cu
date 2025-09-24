#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cfloat>

using namespace std;

typedef vector<vector<float>> Matrix;

// ---------- Transpose Kernel + Host ---------------------
__global__ void transposevecKernel(const float* A, float* B, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        B[col*M+row] = A[row*N+col];
    }
}

Matrix transposevec(const Matrix& A){
    int M = A.size();
    int N = A[0].size();

    vector<float> A_flat(M*N);
    vector<float> B_flat(N*M, 0);

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            A_flat[i*N+j] = A[i][j];
        }
    }
    float *d_a, *d_b;
    size_t size = sizeof(float);

    cudaMalloc(&d_a, M*N*size);
    cudaMalloc(&d_b, N*M*size);

    cudaMemcpy(d_a, A_flat.data(), M*N*size, cudaMemcpyHostToDevice);
    
    //launch kernel
    dim3 threads (16,16);
    dim3 blocks ((N+15)/16,(M+15)/16);

    transposevecKernel<<<blocks, threads>>>(d_a,d_b,M,N);
    cudaDeviceSynchronize();

    cudaMemcpy(B_flat.data(), d_b, N*M*size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);

    Matrix B(A[0].size(), vector<float>(A.size()));
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            B[i][j] = B_flat[i*M+j];
        }
    }

    return B;
}

// ---------- Matmul Kernel + Host ---------------------
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
Matrix matmul(const Matrix& A, const Matrix& B){
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

// ---------- Scale Kernel + Host ---------------------
__global__ void scalematrixKernel(float* A, const float* B, int M, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < M*N){
       A[i] /= sqrtf(*B); 
    }
}
Matrix scalematrix(Matrix& A, float scalar){
    int M = A.size();
    int N = A[0].size();

    vector<float> A_flat(M*N);
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            A_flat[i*N+j] = A[i][j];
        }
    }

    float *d_A, *d_B;
    size_t size = sizeof(float);

    cudaMalloc(&d_A, M*N*size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, A_flat.data(), M*N*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, &scalar, size, cudaMemcpyHostToDevice);

    //launch kernel
    dim3 threads = 256;
    dim3 blocks(((M*N)+255)/256);
    scalematrixKernel<<<blocks, threads>>>(d_A,d_B,M,N);
    cudaDeviceSynchronize();

    cudaMemcpy(A_flat.data(), d_A, M*N*size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            A[i][j] = A_flat[i*N+j];
        }
    }
    return A;
}

// ---------- Softmax Kernel + Host ---------------------
__global__ void softmaxKernel(const float* A, float* B, int M, int N){
    int row = blockIdx.x;

    if(row < M){
        // row max
        float max_val = -FLT_MAX;
        for(int i = 0; i < N; ++i){
            float val = A[row * N + i];
            max_val = max(val, max_val);
        }
        

        //exp
        float sum = 0.0f;
        float temp[512];
        for(int i = 0; i < N; ++i){
            float diff = A[row * N + i] - max_val;
            if (diff < -20.0f) diff = -20.0f; // prevent underflow
            float val = expf(diff);

            // float val = expf(A[row*N+i] - max_val);
            temp[i] = val;
            // B[row*N+i] = val;
            sum += val;
        }
        
        //softmax finale- normalize 
        for(int i = 0; i < N; ++i){
            B[row*N+i] = temp[i] / sum;
        }
    }
}

Matrix softmax(const Matrix& A){
    int M = A.size();
    int N = A[0].size();

    vector<float> A_flat(M*N);
    vector<float> B_flat(M*N,0);

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            A_flat[i*N+j] = A[i][j];
        }
    }

    float *d_A, *d_B;
    size_t size = sizeof(float);

    cudaMalloc(&d_A, M*N*size);
    cudaMalloc(&d_B, M*N*size);

    cudaMemcpy(d_A, A_flat.data(), M*N*size, cudaMemcpyHostToDevice);

    //LAUNCH KERNEL
    int threads = 256;
    int blocks= (M + 255) / 256;
    softmaxKernel<<<blocks, threads>>>(d_A,d_B,M,N);
    cudaDeviceSynchronize();

    cudaMemcpy(B_flat.data(), d_B, M*N*size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    Matrix B(M, vector<float>(N));
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            B[i][j] = B_flat[i*N+j];
        }
    }
    return B;

}

//-------------PRINTER-----------
void printer(const Matrix& Mat, const string& label){
    cout << label << endl;
    for(auto& row : Mat){
        for(auto& val : row){
            cout << val << " ";
        }
        cout << endl;
    }
    cout << endl;
}
// ---------- Attention Host ---------------------
Matrix attention(const Matrix& Q, const Matrix& K, const Matrix& V, int d_k){
    //GET Kt
    auto Kt = transposevec(K);
    printer(Kt, "Transposed K: ");
    //GET Q.Kt
    auto unscaled_similarities = matmul(Q, Kt);
    printer(unscaled_similarities, "Unscaled dot product Similarities: ");
    //SCALE BY SQRT(D_K)
    auto scaled_similarities = scalematrix(unscaled_similarities, d_k);
    printer(scaled_similarities, "Scaled Similarities: ");
    //SOFTMAX
    auto attention_percentages = softmax(scaled_similarities);
    printer(attention_percentages, "Attention percentages: SoftMax scores: ");
    // xV
    auto self_attention = matmul(attention_percentages, V);
    printer(self_attention, "Self Attention scores for this head: ");

    return self_attention;
}

//------------Concatenator Kernel + Host---------
__global__ void concatKernel(const float* head1, const float* head2, float* output, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row<M && col < N){
        int idx = row * N + col;
        int out_idx1 = row *2 * N + col;
        int out_idx2 = row * 2 * N + N + col;

        output[out_idx1] = head1[idx];
        output[out_idx2] = head2[idx];
    }
}
Matrix concat(const Matrix& head1, const Matrix& head2){
    int M = head1.size();
    int N = head2[0].size();

    vector<float> h1_flat(M*N);
    vector<float> h2_flat(M*N);
    vector<float> output_flat(M*(N*2), 0);

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            h1_flat[i*N+j] = head1[i][j];
        }
    }
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            h2_flat[i*N+j] = head2[i][j];
        }
    }

    float *d_h1, *d_h2, *d_out;
    size_t size = sizeof(float);

    cudaMalloc(&d_h1, M*N*size);
    cudaMalloc(&d_h2, M*N*size);
    cudaMalloc(&d_out, size*M*(2*N));

    cudaMemcpy(d_h1, h1_flat.data(), M*N*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h2, h2_flat.data(), M*N*size, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks((N+15)/16, (M+15)/16);
    concatKernel<<<blocks, threads>>>(d_h1,d_h2,d_out,M,N);
    cudaDeviceSynchronize();

    cudaMemcpy(output_flat.data(), d_out, size*M*(2*N), cudaMemcpyDeviceToHost);

    cudaFree(d_h1);
    cudaFree(d_h2);
    cudaFree(d_out);

    Matrix output(M,vector<float>(2*N));
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < (2*N); ++j){
            output[i][j] = output_flat[i*(2*N)+j];
        }
    }
    return output;
}

//----------Multi Head Calling Simulator------
Matrix multi_head_attention(
    const Matrix& Q,
    const Matrix& K,
    const Matrix& V,
    const int& d_k,
    const Matrix& W_q1,
    const Matrix& W_q2,
    const Matrix& W_k1,
    const Matrix& W_k2,
    const Matrix& W_v1,
    const Matrix& W_v2,
    const Matrix& W_o
){
    auto Q1 = matmul(Q, W_q1);
    auto K1 = matmul(K, W_k1);
    auto V1 = matmul(V, W_v1);

    auto Q2 = matmul(Q, W_q2);
    auto K2 = matmul(K, W_k2);
    auto V2 = matmul(V, W_v2);

    auto head1 = attention(Q1, K1, V1, d_k);
    auto head2 = attention(Q2, K2, V2, d_k);

    auto result = concat(head1, head2);
    auto final_linear = matmul(result, W_o);

    return final_linear;
}

//---------------Matrix addition Kernel + Host --------------
__global__ void addMatrixKernel(const float* mat1, const float* mat2, float* output, int M, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < M*N){
        output[idx] = mat1[idx] + mat2[idx];
    }
}
Matrix addMatrix(const Matrix& mat1, const Matrix& mat2){
    int M = mat1.size();
    int N = mat2[0].size();
    int totalsize = M*N;
    vector<float> m1_flat(totalsize);
    vector<float> m2_flat(totalsize);
    vector<float> output_flat(totalsize,0);

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            m1_flat[i*N+j] = mat1[i][j];
        }
    }
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            m2_flat[i*N+j] = mat2[i][j];
        }
    }

    float *d_m1, *d_m2, *d_out;
    size_t size = sizeof(float);

    cudaMalloc(&d_m1, totalsize * size);
    cudaMalloc(&d_m2, totalsize * size);
    cudaMalloc(&d_out, totalsize * size);

    cudaMemcpy(d_m1, m1_flat.data(), totalsize*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, m2_flat.data(), totalsize*size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (totalsize + threads - 1) / threads;
    addMatrixKernel<<<blocks, threads>>>(d_m1,d_m2,d_out,M,N);
    cudaDeviceSynchronize();

    cudaMemcpy(output_flat.data(), d_out, totalsize * size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_out);

    Matrix output(M, vector<float>(N));
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            output[i][j] = output_flat[i*N+j];
        }
    }

    return output;    
}

//-------------addBias Kernel + host --------------
__global__ void addBiasKernel(const float* mat, const float* bias, float* output, int M, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M*N;
    if (idx < total){
        int col = idx % N;
        output[idx] = mat[idx] + bias[col];
    }
}
Matrix addBias(const Matrix& mat, const vector<float>& bias){
    int M = mat.size();
    int N = mat[0].size();

    vector<float> mat_flat(M*N);
    vector<float> output_flat(M*N,0);
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            mat_flat[i*N+j] = mat[i][j];
        }
    }

    float *d_mat, *d_bias, *d_out;
    size_t size = sizeof(float);

    cudaMalloc(&d_mat, M*N*size);
    cudaMalloc(&d_out, M*N*size);
    cudaMalloc(&d_bias, N*size);

    cudaMemcpy(d_mat, mat_flat.data(), M*N*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.data(), N*size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = ((M*N) + 255) / 256;
    addBiasKernel<<<blocks,threads>>>(d_mat,d_bias,d_out,M,N);
    cudaDeviceSynchronize();

    cudaMemcpy(output_flat.data(), d_out, M*N*size, cudaMemcpyDeviceToHost);

    cudaFree(d_mat);
    cudaFree(d_bias);
    cudaFree(d_out);

    Matrix output(M, vector<float>(N));
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            output[i][j] = output_flat[i*N+j];
        }
    }
    return output;
}

//--------------ReLU Kernel + Host-------------
__global__ void reluKernel(float* input, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        if(input[idx] < 0){
            input[idx] = 0;
        }
    }
}
Matrix relu(Matrix& input){
    int M = input.size();
    int N = input[0].size();
    int inp_size = M*N;

    vector<float> in_flat(inp_size);
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            in_flat[i*N+j] = input[i][j]; 
        }
    }

    float *d_in;

    cudaMalloc(&d_in, inp_size*sizeof(float));

    cudaMemcpy(d_in, in_flat.data(), inp_size*sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (inp_size + 255) / 256;
    reluKernel<<<blocks,threads>>>(d_in, inp_size);
    cudaDeviceSynchronize();

    cudaMemcpy(in_flat.data(),d_in, inp_size*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            input[i][j] = in_flat[i*N+j];
        }
    }
    return input;
    
}

//----------FEED FORWARD NETWORK-----------
//FFN(x) = ReLU(x * W1 + b1) * W2 + b2
Matrix FeedForward(const Matrix& mat, const Matrix& w1, const vector<float>& b1, const Matrix& w2, const vector<float>& b2){
    auto linear1 = matmul(mat, w1);
    auto bias_added = addBias(linear1, b1);
    auto rectified = relu(bias_added);
    auto linear2 = matmul(rectified,w2);
    auto result = addBias(linear2, b2);
    return result;
}
//-------------Layer Norm Kernel + Host-----------
__global__ void layerNormKernel(const float* input, float* output, int M, int N, float epsilon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // mean for the row
    float sum = 0.0f;
    for (int col = 0; col < N; ++col) {
        sum += input[row * N + col];
    }
    float mean = sum / N;

    // variance for the row
    float var_sum = 0.0f;
    for (int col = 0; col < N; ++col) {
        float val = input[row * N + col];
        var_sum += (val - mean) * (val - mean);
    }
    float variance = var_sum / N;
    float denom = sqrtf(variance + epsilon);

    // normalize each element in the row
    for (int col = 0; col < N; ++col) {
        output[row * N + col] = (input[row * N + col] - mean) / denom;
    }
}
Matrix layerNorm(const Matrix& A){
    int M = A.size();
    int N = A[0].size();
    float epsilon=1e-5;

    vector<float> A_flat(M*N);
    vector<float> B_flat(M*N,0);

    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            A_flat[i*N+j] = A[i][j];
        }
    }

    float *d_A, *d_B;
    size_t size = sizeof(float);

    cudaMalloc(&d_A, M*N*size);
    cudaMalloc(&d_B, M*N*size);

    cudaMemcpy(d_A, A_flat.data(), M*N*size, cudaMemcpyHostToDevice);

    //LAUNCH KERNEL
    int threads = 256;
    int blocks= (M + 255) / 256;
    layerNormKernel<<<blocks, threads>>>(d_A,d_B,M,N,epsilon);
    cudaDeviceSynchronize();

    cudaMemcpy(B_flat.data(), d_B, M*N*size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    Matrix B(M, vector<float>(N));
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            B[i][j] = B_flat[i*N+j];
        }
    }
    return B;

}
int main() {
    int seq_len = 4;
    int model_dim = 8;
    int d_k = 4;

    // Simulated input
    Matrix Q = {
    {0.5, 1.2, 0.7, 0.9, 1.1, 0.3, 0.8, 0.6},
    {1.0, 1.1, 0.6, 1.3, 1.0, 0.4, 0.7, 0.5},
    {0.6, 0.9, 1.2, 1.0, 1.3, 0.5, 0.9, 0.4},
    {0.7, 1.0, 0.8, 1.1, 0.9, 0.6, 1.1, 0.7}
    };
    Matrix K = Q;
    Matrix V = Q;

    // Randomly initialized weights
    Matrix W_Q1 = {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1},
        {1, 1, 0, 0},
        {0, 1, 1, 0},
        {0, 0, 1, 1},
        {1, 0, 0, 1}
    };


    Matrix W_K1 = {
        {0.5, 0.5, 0, 0},
        {0, 0.5, 0.5, 0},
        {0, 0, 0.5, 0.5},
        {0.5, 0, 0, 0.5},
        {0.25, 0.25, 0.25, 0.25},
        {0.6, 0.2, 0.1, 0.1},
        {0.1, 0.6, 0.2, 0.1},
        {0.1, 0.1, 0.6, 0.2}
    };

    Matrix W_V1 = {
        {1, 1, 1, 1},
        {2, 2, 2, 2},
        {3, 3, 3, 3},
        {4, 4, 4, 4},
        {5, 5, 5, 5},
        {6, 6, 6, 6},
        {7, 7, 7, 7},
        {8, 8, 8, 8}
    };

    Matrix W_Q2 = {
        {0.1, 0.2, 0.3, 0.4},
        {0.2, 0.3, 0.4, 0.5},
        {0.3, 0.4, 0.5, 0.6},
        {0.4, 0.5, 0.6, 0.7},
        {0.5, 0.6, 0.7, 0.8},
        {0.6, 0.7, 0.8, 0.9},
        {0.7, 0.8, 0.9, 1.0},
        {0.8, 0.9, 1.0, 1.1}
    };

    Matrix W_K2 = {
        {0.4, 0.3, 0.2, 0.1},
        {0.5, 0.4, 0.3, 0.2},
        {0.6, 0.5, 0.4, 0.3},
        {0.7, 0.6, 0.5, 0.4},
        {0.8, 0.7, 0.6, 0.5},
        {0.9, 0.8, 0.7, 0.6},
        {1.0, 0.9, 0.8, 0.7},
        {1.1, 1.0, 0.9, 0.8}
    };

    Matrix W_V2 = {
        {0.1, 0.1, 0.1, 0.1},
        {0.2, 0.2, 0.2, 0.2},
        {0.3, 0.3, 0.3, 0.3},
        {0.4, 0.4, 0.4, 0.4},
        {0.5, 0.5, 0.5, 0.5},
        {0.6, 0.6, 0.6, 0.6},
        {0.7, 0.7, 0.7, 0.7},
        {0.8, 0.8, 0.8, 0.8}
    };

    Matrix W_o = {
        {1, 0, 0, 0, 1, 0, 0, 0},
        {0, 1, 0, 0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0, 0, 1, 0},
        {0, 0, 0, 1, 0, 0, 0, 1},
        {0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0},
        {0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0},
        {0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5},
        {0.5, 0, 0, 0.5, 0.5, 0, 0, 0.5}
    };
    
    auto result = multi_head_attention(Q,K,V,d_k,W_Q1,W_Q2,W_K1,W_K2,W_V1,W_V2,W_o);
    cout << "FINAL MULTI HEAD ATTENTION!!!\n" << "DRUM ROLLS\n";
    printer(result, "Multi Head Attention Scores Concatenated: ");
    return 0;
}
