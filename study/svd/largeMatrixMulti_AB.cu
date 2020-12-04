#include <iostream>
#include <fstream>
#include <assert.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <math.h>
#include <curand.h>
#include <cufft.h>
#include <cuda_fp16.h>

using namespace std;
typedef float dt;
bool reduceDim =true;
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
const float alpha = 1.0, beta0 = 0.0, beta_1 = -1.0, beta1 = 1.0;

__global__ void warmup()
{
    int sum=0;
    for(int i = 0; i < 1000; i++) {
        sum+=i;
    }
}   

void warmupcu(){
    warmup<<<1,1>>>();
}

void norm2HH(float *A, long long len, float *norm2){
  // printf("%lld\n", len);
  double norm2Result = 0.0;
  for(long long i = 0; i < len; ++i){
    norm2Result += (A[i] * A[i]);
  }
  *norm2 = sqrtf(norm2Result);
}

void norm2HH_device(float *d_A, long long len, float *norm2){
    // printf("%lld\n", len);
    dt *h_A = NULL;
    cudaHostAlloc((void**)&h_A , sizeof(dt)*len, 1);
    cudaMemcpy(h_A, d_A, sizeof(dt)*len, cudaMemcpyDeviceToHost);
    norm2HH(h_A, len, norm2);
    if(h_A) cudaFreeHost(h_A);
}


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
        return time*1e-3;
    }
    private:
    cudaEvent_t start_, stop_;
};

// /usr/local/cuda/bin/nvcc -lcublas -lcusolver -lcurand -std=c++11 largeMatrixMulti_AB.cu -o largeMatrixMulti_AB
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
// nvcc -lcublas -lcusolver -lcurand -std=c++11 svd.cu -o svd
void printMatrix_Device(int m, int n, const dt*d_A, int lda, const char* name)
{
    dt *h_A = NULL;
    cudaHostAlloc((void**)&h_A , sizeof(dt)*m*n, 1);
    cudaMemcpy(h_A, d_A, sizeof(dt)*m*n, cudaMemcpyDeviceToHost);
    printMatrix(m, n, h_A, lda, name);
    if(h_A) cudaFreeHost(h_A);
}

void printMatrix_half(int m, int n, const half*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = (float)A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
    cout << " ------------------------------------" << endl;
}
// nvcc -lcublas -lcusolver -lcurand -std=c++11 svd.cu -o svd
void printMatrix_Device_half(int m, int n, const half*d_A, int lda, const char* name)
{
    half *h_A = NULL;
    cudaHostAlloc((void**)&h_A , sizeof(half)*m*n, 1);
    cudaMemcpy(h_A, d_A, sizeof(half)*m*n, cudaMemcpyDeviceToHost);
    printMatrix_half(m, n, h_A, lda, name);
    if(h_A) cudaFreeHost(h_A);
}




__global__  void floattohalf(dt *AA,half *BB,long m){
    long i = blockIdx.x*blockDim.x+threadIdx.x;
    const long temp = blockDim.x*gridDim.x;
    if(i<m){
        BB[i] = __float2half(AA[i]);
        i+=temp;
    }
    __syncthreads();
}

void f2h(dt *A,half *B,long num){
    dim3 threads(1024,1,1);
    dim3 blocks((num+1024-1)/1024,1,1);   
    floattohalf<<<blocks,threads>>>(A,B,num);
}

void smallargeMatrixMulti_slice_host(cublasHandle_t cublasH, dt *d_A, dt *h_B, const int &ttrank, const int &m,  const long long &n, dt *h_ATB, const int &p){
    long long slice = 0;
    if(n%p==0)
        slice = n/p;
    else
        assert("P is not slice");
    // printf("ttrank: %d, n: %d",m,ttrank,n);
    float *d_tempB = NULL, *d_tempATB = NULL; 
    half *d_Ah = NULL, *d_Bh = NULL;
    cudaStat1 = cudaMalloc((void**)&d_tempB, sizeof(dt)*m*slice);
    cudaStat2 = cudaMalloc((void**)&d_tempATB, sizeof(dt)*ttrank*slice);
    cudaStat3 = cudaMalloc((void**)&d_Ah, sizeof(half)*m*ttrank);
    cudaStat4 = cudaMalloc((void**)&d_Bh, sizeof(half)*m*slice);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);

    // printMatrix_Device(m, ttrank, d_A, m, "tempA");
    f2h(d_A, d_Ah, m*ttrank);
    // printf("%d\n", slice);
    for(int i=0;i<p;i++){
        cudaStat3 = cudaMemcpyAsync(d_tempB, h_B+m*slice*i, sizeof(dt)*m*slice, cudaMemcpyHostToDevice,0);
        cudaStat4 = cudaDeviceSynchronize();
        assert(cudaStat3 == cudaSuccess);
        assert(cudaStat4 == cudaSuccess);
        // printMatrix_Device(m, slice, d_tempB, m, "tempB");
        f2h(d_tempB, d_Bh, m*slice);
        cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                           ttrank, slice, m,
                           &alpha, d_Ah, CUDA_R_16F, m,
                           d_Bh, CUDA_R_16F, m,
                           &beta0, d_tempATB, CUDA_R_32F, ttrank,
                           CUDA_R_32F, algo);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);
        // printMatrix_Device(ttrank, slice, d_tempATB, ttrank, "tempATB");

        cudaStat3 = cudaMemcpyAsync(h_ATB+ttrank*slice*i, d_tempATB, sizeof(dt)*ttrank*slice, cudaMemcpyDeviceToHost, 0);
        cudaStat4 = cudaDeviceSynchronize();
        assert(cudaStat3 == cudaSuccess);
        assert(cudaStat4 == cudaSuccess);
    }
    if(d_Ah     ) cudaFree(d_Ah); d_Ah=NULL;
    if(d_Bh     ) cudaFree(d_Bh); d_Bh=NULL;
    if(d_tempB  ) cudaFree(d_tempB); d_tempB=NULL;
    if(d_tempATB  ) cudaFree(d_tempATB); d_tempATB=NULL;
}


void smallargeMatrixMulti_slice_device(cublasHandle_t cublasH, dt *d_A, dt *d_B, const int &ttrank, const int &m, const long long &n, dt *d_ATB, const int &p){
    half *d_Ah = NULL, *d_Bh = NULL;

    long long slice = n/p;
    cudaStat1 = cudaMalloc((void**)&d_Ah, sizeof(half)*m*ttrank);
    cudaStat2 = cudaMalloc((void**)&d_Bh, sizeof(half)*m*n);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    f2h(d_A, d_Ah, m*ttrank);
    f2h(d_B, d_Bh, m*n);
    //slice by slice
    for(int i = 0; i<p; i++){

        cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                           ttrank, slice, m,
                           &alpha, d_Ah, CUDA_R_16F, m,
                           d_Bh+i*m*slice, CUDA_R_16F, m,
                           &beta0, d_ATB+slice*i*ttrank, CUDA_R_32F, ttrank,
                           CUDA_R_32F, algo);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    }
    if(d_Ah     ) cudaFree(d_Ah); d_Ah=NULL;
    if(d_Bh     ) cudaFree(d_Bh); d_Bh=NULL;
}

int main(){
    cublasHandle_t cublasH = NULL;
    const int m = 4;
    const int n = 40;
    const int ttRank = 2;
   
    printf("************************ks************************\n %d \n*****************************************\n",m);
    dt *h_A = NULL,*h_B = NULL, *d_A = NULL, *d_A2=NULL, *d_B=NULL, *h_A2=NULL;
    cudaHostAlloc((void**)&h_A,sizeof(dt)*m*ttRank,0);
    cudaHostAlloc((void**)&h_B,sizeof(dt)*n*m,0);
    for(int i = 0; i < ttRank*m; i++){
        h_A[i] = (dt) rand()*1.0 / RAND_MAX*1.0*2.0 - 1.0;
    }
    for(int i = 0; i < n*m; i++){
        h_B[i] = (dt) rand()*1.0 / RAND_MAX*1.0*2.0 - 1.0;
    }

    cudaStat1 = cudaMalloc((void**)&d_A, sizeof(dt)*ttRank*m);
    cudaStat2 = cudaMalloc((void**)&d_B, sizeof(dt)*n*m);
    cudaStat3 = cudaMalloc((void**)&d_A2, sizeof(dt)*ttRank*n);
    cudaStat4 = cudaHostAlloc((void**)&h_A2, sizeof(dt)*ttRank*n,0);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cublas_status = cublasSetMathMode(cublasH, mathMode);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cudaStat1 = cudaMemcpyAsync(d_A, h_A, sizeof(dt)*ttRank*m, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpyAsync(d_B, h_B, sizeof(dt)*m*n, cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpyAsync(d_A2, d_A2, sizeof(dt)*ttRank*n, cudaMemcpyHostToDevice);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);

    printMatrix_Device(m,ttRank,d_A,m,"Ay");
    printMatrix_Device(m,n,d_B,m,"By");
    smallargeMatrixMulti_slice_device(cublasH, d_A, d_B, ttRank, m, n, d_A2, 2);
    printMatrix_Device(ttRank,n,d_A2,ttRank,"A2");
    smallargeMatrixMulti_slice_host(cublasH, d_A, h_B, ttRank, m, n, h_A2, 2);
    printMatrix(ttRank,n,h_A2,ttRank,"A22");
    if(cublasH ) cublasDestroy(cublasH);
    if(h_A     ) cudaFreeHost(h_A); h_A = NULL;
    if(h_A2    ) cudaFreeHost(h_A2); h_A2 = NULL;
    if(h_B     ) cudaFreeHost(h_B); h_B = NULL;
    if(d_A     ) cudaFree(d_A); d_A = NULL;
    if(d_B    ) cudaFree(d_B); d_B = NULL;
    if(d_A2   ) cudaFree(d_A2); d_A2 = NULL;
    // cudaDeviceReset();
    return 0;
}