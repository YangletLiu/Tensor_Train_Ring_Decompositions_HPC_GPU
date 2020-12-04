/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include syevd_example.cpp 
     /usr/local/cuda/bin/nvcc -lcublas -lcusolver -lcurand -std=c++11 eign.cu -o eign
 *   g++ -o a.out syevd_example.o -L/usr/local/cuda/lib64 -lcudart -lcusolver
 *
 */

#include <iostream>
#include <fstream>
#include <assert.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <math.h>
#include <curand.h>
#include <cufft.h>

using namespace std;
typedef float dt;
cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP; //CUBLAS_GEMM_DEFAULT_TENSOR_OP CUBLAS_GEMM_DEFAULT
cublasMath_t mathMode = CUBLAS_TENSOR_OP_MATH; //CUBLAS_TENSOR_OP_MATH  CUBLAS_DEFAULT_MATH
cudaError_t cudaStat1 = cudaSuccess;
cudaError_t cudaStat2 = cudaSuccess;
cudaError_t cudaStat3 = cudaSuccess;
cudaError_t cudaStat4 = cudaSuccess;
cudaError_t cudaStat5 = cudaSuccess;
cudaError_t cudaStat6 = cudaSuccess;
const dt alpha = 1.0, beta = 0.0;


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

void genTTTensor(dt *T,long a,long b,long c,long r){
    dt *AA,*BB,*CC;    
    cudaHostAlloc((void**)&AA,sizeof(dt)*a*r,0);
    cudaHostAlloc((void**)&BB,sizeof(dt)*b*r,0);
    cudaHostAlloc((void**)&CC,sizeof(dt)*c*r,0);
    for(long i = 0;i<a*r;i++){
        AA[i]=rand()*0.1/(RAND_MAX*0.1);
    }
    for(long i = 0;i<b*r;i++){
        BB[i]=rand()*0.1/(RAND_MAX*0.1);
    }
    for(long i = 0;i<c*r;i++){
        CC[i]=rand()*0.1/(RAND_MAX*0.1);
    }
    dt *d_T,*d_CC,*d_BB,*d_AA;
    cudaMalloc((void**)&d_AA,sizeof(dt)*a*r);
    cudaMalloc((void**)&d_BB,sizeof(dt)*b*r);
    cudaMalloc((void**)&d_CC,sizeof(dt)*c*r);
    cudaMalloc((void**)&d_T,sizeof(dt)*a*b*c);
    cudaMemcpyAsync(d_BB,BB,sizeof(dt)*b*r,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_CC,CC,sizeof(dt)*c*r,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_AA,AA,sizeof(dt)*a*r,cudaMemcpyHostToDevice,0);
    dt *d_CKRB;
    cudaMalloc((void**)&d_CKRB,sizeof(dt)*c*r*b);
    cudaDeviceSynchronize();

    // printMatrix_Device(a,r,d_AA,a,"AA");
    // printMatrix_Device(b,r,d_BB,b,"BB");
    // printMatrix_Device(c,r,d_CC,c,"CC");

    //X1=A*(CkrB)'  a*r  r*(bc)
    dt alpha = 1.0;
    dt beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,
        b,c,1,
        &alpha,
        d_BB,b,b,
        d_CC,c,c,
        &beta,d_CKRB,
        b,b*c,r);
    // printMatrix_Device(b,c*r,d_CKRB,b,"CkrB");

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b*c,r,&alpha,d_AA,a,d_CKRB,b*c,&beta,d_T,a);
    // printMatrix_Device(a,b*c,d_T,a,"T");

    cudaMemcpyAsync(T,d_T,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost,0);
    cudaDeviceSynchronize();

    cudaFree(d_AA);
    cudaFree(d_BB);
    cudaFree(d_CC);
    cudaFree(d_CKRB);
    cudaFree(d_T);
    cudaFreeHost(AA);
    cudaFreeHost(BB);
    cudaFreeHost(CC);
    cublasDestroy(handle);
}

__global__ void matrixInvertColumn(dt *d_A, dt *d_Ainv, int m, int n){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*n){
        int row = i%m;
        int col = i/m;
        d_Ainv[row+(n-col-1)*m]=d_A[i];
        // d_Ainv[i] = d_A[];
    }
    __syncthreads();
}

void printMatrix(int m, int n, const dt*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            dt Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

void printMatrix_Device(int m, int n, const dt*d_A, int lda, const char* name)
{
    dt *h_A = NULL;
    cudaHostAlloc((void**)&h_A , sizeof(dt)*m*n, 1);
    cudaMemcpy(h_A, d_A, sizeof(dt)*m*n, cudaMemcpyDeviceToHost);
    printMatrix(m, n, h_A, lda, name);
    if(h_A) cudaFreeHost(h_A);
}

//在TT分解中返回的是U 然后利用
void basicEig(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, dt *d_A, const int m, dt *d_W){  
    int *devInfo = NULL;
    dt *d_work = NULL;
    int  lwork = 0;
    int info_gpu = 0;

    cudaMalloc ((void**)&devInfo, sizeof(int));
    
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnSsyevd_bufferSize(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A,
        m,
        d_W,
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(dt)*lwork);
    assert(cudaSuccess == cudaStat1);

// step 4: compute spectrum
    cusolver_status = cusolverDnSsyevd(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A,
        m,
        d_W,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("after syevd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    // if (d_W    ) cudaFree(d_W);
}

int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    const int m = 160;
    const int n = 1600*1600;
    // const int r = 50;
    // int r_mul8 = 56;  //Multiple of 8
    // if(m > 500){
    //     r_mul8 = q/10/8*8;
    // }
    // genTTTensor(A,m,m,m,r);
    dt *A=NULL;
    cudaMallocHost((void**)&A, sizeof(dt)*m*n);
    for(int i = 0; i < n*m; i++){
        A[i] = (dt) (rand()*1.0 / RAND_MAX*1.0)*2.0-1.0;
    }
    // dt V[n*m]; // eigenvectors
    // dt W[m]; // eigenvalues

    dt *d_U = NULL;
    dt *d_A = NULL;
    dt *d_W = NULL;
    dt *d_A2 = NULL;
    dt *d_Ueig = NULL;
    
// step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cublas_status = cublasSetMathMode(cublasH, mathMode);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

// step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(dt) * m * n);
    cudaStat1 = cudaMalloc ((void**)&d_Ueig, sizeof(dt) * m * m);
    cudaStat1 = cudaMalloc ((void**)&d_U, sizeof(dt) * m * m);
    cudaStat1 = cudaMalloc ((void**)&d_A2, sizeof(dt) * m * n);
    cudaStat2 = cudaMalloc ((void**)&d_W, sizeof(dt) * m);
    
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(dt) * m * n, cudaMemcpyHostToDevice);
    // printMatrix_Device(m, n, d_A, m, "A");
    assert(cudaSuccess == cudaStat1);

    GPUTimer timer0;
    timer0.start();
    cublas_status = cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, m, m, n, &alpha, d_A, m, d_A, m, &beta, d_Ueig, m);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    printMatrix_Device(m, m, d_Ueig, m, "AAT");

    basicEig(cusolverH, cublasH, d_Ueig, m, d_W);
    if (d_W) cudaFree(d_W);
    // printMatrix_Device(m, m, d_Ueig, m, "Ueig");
    dim3 threads(512,1,1);
    dim3 blocksUinv((m*m+512-1)/512,1,1);
    matrixInvertColumn<<<blocksUinv, threads>>>(d_Ueig, d_U, m, m);
    if(d_Ueig ) cudaFree(d_Ueig);
    // printMatrix_Device(m, m, d_U, m, "U");
    // printf("=====\n");
 
    cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                           m, n, m,
                           &alpha, d_U, CUDA_R_32F, m,
                           d_A, CUDA_R_32F, m,
                           &beta, d_A2, CUDA_R_32F, m,
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    float time2 = timer0.seconds();
    printf("\n***********************\n time1: %f \n*********************\n", time2);

    // printMatrix_Device(m, n, d_A2, m, "SVT");
    // printf("=====\n");
    
    if (d_A    ) cudaFree(d_A);
    if (d_A2  ) cudaFree(d_A2);
    if (d_U    ) cudaFree(d_U);
    if (A) cudaFreeHost(A);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (cublasH ) cublasDestroy(cublasH);
    cudaDeviceReset();

    return 0;
}