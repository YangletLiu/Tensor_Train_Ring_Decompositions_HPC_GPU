#include <iostream>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <time.h>
#include <fstream>
using namespace std;
typedef float lg;
typedef float dt;


struct GPUTimer
{
    GPUTimer() 
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~GPUTimer() 
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() 
    {
        cudaEventRecord(start_, 0);
    }

    float seconds() 
    {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time ;
    }
    private:
    cudaEvent_t start_, stop_;
};

// nvcc -lcublas -lcusolver -lcurand -std=c++11 svd.cu -o svd
void printMatrix(int m, int n, const dt*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
        	// if(row == col){
            	float Areg = A[row + col*lda];
            	printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        	// }
        }
    }
    cout << " ------------------------------------" << endl;
}
// nvcc -lcublas -lcusolver -lcurand -std=c++11 svd.cu -o svd
void printMatrix_Device(int m, int n, const dt*d_A, int lda, const char* name)
{
    dt *h_A = NULL;
    cudaHostAlloc((void**)&h_A , sizeof(dt)*m*n, 1);
    cudaMemcpy(h_A, d_A, sizeof(dt)*m*n, cudaMemcpyDeviceToHost);
    printMatrix(m, n, h_A, lda, name);
    if(h_A) cudaFreeHost(h_A);
}


__global__ void tranforArrayToDiagonalMatrix(float* array, float* diagMatrix, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int temp = blockDim.x*gridDim.x;
	while(i < n){
		diagMatrix[n*i+i] = array[i];
		i+=temp;
	}
	__syncthreads();
}

// __global__ void krpro(float *A,float *B,float *C,lg m,lg k,lg r)
// {
// 	lg i = blockIdx.x * blockDim.x + threadIdx.x;
// 	const lg temp = blockDim.x*gridDim.x;

// 	while(i<m*k*r)
// 	{
// 		lg row = i%(m*k);
// 		lg col = i/(m*k);
// 		C[i]=A[i/k]*B[col*k+i%k];
// 		//C[col*m*k+row] = A[(row/k)+col*m]*B[(row%k)+col*k];
// 		i+=temp;
// 	}
// 	__syncthreads();
// }


int main(int argc, char const *argv[]) {
	cublasStatus_t  cublas_status = CUBLAS_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cublasHandle_t cublasH;
	cublasCreate(&cublasH);
	cublasSetMathMode(cublasH,CUBLAS_TENSOR_OP_MATH);
	// cublasSetMathMode(cublasH,CUBLAS_DEFAULT_MATH);

//1630，1640，1650
	int n = 1200; //第一个矩阵的row  kr pro 的列数是一样的  第一个m*r
	int m = n*n;
	float *S = NULL;
	float *U = NULL;
	cudaHostAlloc((void**)&S,sizeof(float)*n,0);
	cudaHostAlloc((void**)&U,sizeof(float)*n*m,0);
	srand(6);
	for(int i = 0; i < n; i++) 
		S[i]= (float)rand()*0.1/(RAND_MAX*0.1);
	for(int i = 0; i < n*m; i++) 
		U[i]= (float)rand()*0.1/(RAND_MAX*0.1);

	float alpha = 1.0;
	float beta = 0.0;
	dim3 thread(512,1,1);
	dim3 blocks((n*n+1024-1)/1024,1,1);

	float *d_S;
	float *d_Smatrix;
	float *d_AT2;
	float *d_AT;
	float *d_UT;
	float *d_U;
	cudaMalloc((void**)&d_S,sizeof(float)*n);
	cudaMalloc((void**)&d_Smatrix,sizeof(float)*n*n);
	cudaMalloc((void**)&d_U,sizeof(float)*n*m);
	cudaMalloc((void**)&d_UT,sizeof(float)*n*m);
	cudaMalloc((void**)&d_AT2,sizeof(float)*n*m);
	cudaMalloc((void**)&d_AT,sizeof(float)*n*m);
	cudaMemcpyAsync(d_S,S, sizeof(float)*n, cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_U,U, sizeof(float)*n*m, cudaMemcpyHostToDevice,0);

	cudaDeviceSynchronize();

	// printMatrix_Device(n,1,d_S,n,"S");
	// printMatrix_Device(n,m,d_U,n,"U");
	// printMatrix_Device(n,n,d_B,n,"B");

	float time=0.0f;
	GPUTimer timer;
	timer.start();
	tranforArrayToDiagonalMatrix<<<blocks,thread>>>(d_S,d_Smatrix,n);
	// printMatrix_Device(n,n,d_Smatrix,n,"Smatrix");
	cublas_status = cublasGemmEx(cublasH, 
        CUBLAS_OP_N, CUBLAS_OP_T,
        n, m, n,
        &alpha,
        d_Smatrix, CUDA_R_32F, n,
        d_U, CUDA_R_32F, m,
        &beta,
        d_AT2, CUDA_R_32F, n,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
    time = timer.seconds();
	printf("cost time ex is :%f ms.\n",time);
	// printMatrix_Device(n,n,d_B,n,"B");
	// printMatrix_Device(n,m,d_AT2,n,"AT1");

	GPUTimer timer2;
    timer2.start();
//首先转置 n*econ_s 到 econ_s*n
    cublas_status = cublasSgeam(cublasH,
                                CUBLAS_OP_T, CUBLAS_OP_N, 
                                n, m,
                                &alpha,
                                d_U, m,
                                &beta,
                                d_U, n,
                                d_UT, n);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaDeviceSynchronize();

//再计算矩阵  列向量乘法  econ_s*econ_s  econ_s*n
    cublas_status = cublasSdgmm(cublasH,
                                CUBLAS_SIDE_LEFT,
                                n, m,
                                d_UT, n,
                                d_S, 1,
                                d_AT, n);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaDeviceSynchronize();

    // printMatrix_Device(n,m,d_AT,n,"AT");
    time = timer2.seconds();
    cout << "----------------------------------\npart two gemm run time: " << time << endl;

	cudaFree(d_S);
	cudaFree(d_Smatrix);
	cudaFree(d_U);
	cudaFree(d_UT);
	cudaFree(d_AT2);
	cudaFree(d_AT);
	// cudaFree(d_Smatrix);
	cudaFreeHost(S);
	cudaFreeHost(U);
	return 0;
}	
