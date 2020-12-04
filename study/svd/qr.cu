#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

using namespace std;
cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;    
cudaError_t cudaStat1 = cudaSuccess;
cudaError_t cudaStat2 = cudaSuccess;
cudaError_t cudaStat3 = cudaSuccess;
cudaError_t cudaStat4 = cudaSuccess;
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
        return time * 1e-3 ;
    }
    private:
    cudaEvent_t start_, stop_;
};


// /usr/local/cuda/bin/nvcc -lcublas -lcusolver -lcurand -std=c++11 qr.cu -o qr
void printMatrix(int m, int n, const dt*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
    cout << " ------------------------------------" << endl;
}

void printMatrix_Device(int m, int n, const dt*d_A, int lda, const char* name)
{
    dt *h_A = NULL;
    cudaHostAlloc((void**)&h_A , sizeof(dt)*m*n, 1);
    cudaMemcpy(h_A, d_A, sizeof(dt)*m*n, cudaMemcpyDeviceToHost);
    printMatrix(m, n, h_A, lda, name);
    if(h_A) cudaFreeHost(h_A);
}



void basicQR(cusolverDnHandle_t cusolverH, float *d_A, int m, int n){
    printf("start qr for m: %d, n:%d\n", m,n);
    float *d_work = NULL, *d_tau = NULL;
    int *devInfo = NULL;
    int  lwork = 0; 
    int info_gpu = 0;
    cudaStat1 = cudaMalloc((void**)&d_tau, sizeof(float)*n);
    cudaStat2 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    cusolver_status = cusolverDnSgeqrf_bufferSize(
        cusolverH, 
        m, 
        n, 
        d_A, 
        m, 
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cout << "qr space"<< sizeof(float)*lwork/1024/1024/1024<<"GB"<<endl;
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    assert(cudaSuccess == cudaStat1);
    // assert(cudaSuccess == cudaStat2);

    cusolver_status = cusolverDnSgeqrf(
        cusolverH, 
        m, 
        n, 
        d_A, 
        m, 
        d_tau, 
        d_work, 
        lwork, 
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    // check if QR is good or not
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    // printf("after geqrf: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

    cusolver_status= cusolverDnSorgqr(
        cusolverH,
        m,
        n,
        n,
        d_A,
        m,
        d_tau,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    // printf("after orgqr: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

    if (d_work) cudaFree(d_work); d_work = NULL;
    if (devInfo) cudaFree(devInfo); devInfo = NULL;
    if (d_tau) cudaFree(d_tau); d_tau = NULL;
}


// __global__ getMatrixUTriangular(int n, float* d_A, float* d_R){

// }

int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
   
    const int m = 2360*2360;
    const int n = 240;
/*       | 1 2 3 |
 *   A = | 4 5 6 |
 *       | 2 1 1 |
 *
 *   x = (1 1 1)'
 *   b = (6 15 4)'
 */

    float *A = NULL;
    srand(time(NULL));
    //float A[lda*n] = { 0.4, 0.2, 0.1, 0.5, 0.3, 0.2};
    cudaHostAlloc((void**)&A, sizeof(float)*m*n,0);
    for(int i = 0; i < m*n; ++i){
        A[i] = (float) rand()*1.0 / RAND_MAX*1.0;
    }
 
// step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    
    float *d_A = NULL; // linear memory of GPU  
// step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(float) * m * n);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);

    basicQR(cusolverH, d_A, m, n);

// free resources
    if (d_A    ) cudaFree(d_A);
    if (cusolverH) cusolverDnDestroy(cusolverH);   

    cudaDeviceReset();

    return 0;
}