#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

__global__ void matrixMulKernel(const int* A, const int* B, int* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

vector<vector<int>> readMatrix(const string& filename, int size) {
    vector<vector<int>> matrix(size, vector<int>(size));
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file >> matrix[i][j];
        }
    }
    file.close();
    return matrix;
}

void writeResult(const vector<vector<int>>& matrix, long long time, long long operations, const string& filename) {
    ofstream file(filename);
    file << "Результирующая матрица:\n";
    for (const auto& row : matrix) {
        for (int val : row) {
            file << setw(5) << val << " ";
        }
        file << "\n";
    }
    file << "\n";
    file << "Время выполнения (CUDA): " << time << " мкс (" << time / 1000000.0 << " с)\n";
    file << "Объем задачи: " << operations << " операций\n";
    file << "Размер матрицы: " << matrix.size() << "x" << matrix.size() << "\n";
}

int main() {
    system("chcp 65001 > nul");

    int size;
    cout << "Введите размер матрицы: ";
    cin >> size;

    string base = "C:/Users/gayvo/cuda/";

    auto A = readMatrix(base + "matrix_a.txt", size);
    auto B = readMatrix(base + "matrix_b.txt", size);

    size_t bytes = size * size * sizeof(int);
    
    int *d_A, *d_B, *d_C;
    
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    vector<int> h_A(size * size);
    vector<int> h_B(size * size);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            h_A[i * size + j] = A[i][j];
            h_B[i * size + j] = B[i][j];
        }
    }
    
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 16;
    dim3 blockSize(threadsPerBlock, threadsPerBlock);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x,
                  (size + blockSize.y - 1) / blockSize.y);
 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
   
    cudaEventRecord(start);
    
    matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, size);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    long long durationMicroseconds = static_cast<long long>(milliseconds * 1000);
  
    vector<int> h_C(size * size);
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    
    vector<vector<int>> C(size, vector<int>(size));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = h_C[i * size + j];
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    long long operations = 2LL * size * size * size;
    
    writeResult(C, durationMicroseconds, operations, base + "result_cuda.txt");
    
    cout << "Результат сохранен в " << base << "result_cuda.txt\n";
    cout << "Время выполнения (CUDA ядро): " << durationMicroseconds << " мкс (" 
         << durationMicroseconds / 1000000.0 << " с)\n";
    cout << "Объем задачи: " << operations << " операций\n";
    cout << "Размер матрицы: " << size << "x" << size << "\n";
    
    return 0;
}