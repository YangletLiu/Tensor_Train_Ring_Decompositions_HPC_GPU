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
const float alpha = 1.0, beta = 0.0, beta2 = -1.0;

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

// /usr/local/cuda/bin/nvcc -lcublas -lcusolver -lcurand -std=c++11 largeMatrixMulti_AAT.cu -o largeMatrixMulti_AAT
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

int main(){
    cublasHandle_t cublasH = NULL;
    const int m = 2;
    const int n = 4;
    int calculateTime = 100;
    bool calculateError = true;
    if(calculateError)
        calculateTime=1;
    printf("************************m************************\n %d \n*****************************************\n",m);
    dt *h_A = NULL, *d_A = NULL, *d_AAT=NULL;
    half *d_Ah=NULL,*d_Ah2=NULL;
    cudaHostAlloc((void**)&h_A,sizeof(dt)*m*n,0);
    for(int i = 0; i < n*m; i++){
        h_A[i] = (dt) rand()*1.0 / RAND_MAX*1.0*2.0 - 1.0;
    }

    cudaStat1 = cudaMalloc((void**)&d_A, sizeof(dt)*m*n);
    cudaStat2 = cudaMalloc((void**)&d_Ah, sizeof(half)*m*n);
    cudaStat2 = cudaMalloc((void**)&d_Ah2, sizeof(half)*m*n);
    cudaStat3 = cudaMalloc((void**)&d_AAT, sizeof(dt)*m*m);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cublas_status = cublasSetMathMode(cublasH, mathMode);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cudaStat1 = cudaMemcpyAsync(d_A, h_A, sizeof(dt)*m*n, cudaMemcpyHostToDevice);
    assert(cudaStat1 == cudaSuccess);

    // printMatrix_Device(m,n,d_A,m,"A");   
    warmupcu();

    GPUTimer timer0;
    timer0.start();
    // cublas_status = cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
    //                             n, m, 
    //                             &alpha, d_A, m,
    //                             &beta, d_A, n,
    //                             d_AT, n);
    // assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    float time0 = timer0.seconds();
    printf("*************Transpose time ****************\n %f \n*******************************\n",time0);
    
    for(int i=0;i<calculateTime;i++){
        cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                               m, m, n,
                               &alpha, d_A, CUDA_R_32F, m,
                               d_A, CUDA_R_32F, m,
                               &beta, d_AAT, CUDA_R_32F, m,
                               CUDA_R_32F, algo);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    }
    
    
    float time1 = timer0.seconds();
    float time_cal1 = (timer0.seconds()-time0)/(dt)calculateTime + time0;
    printf("*************Time1 calculate1 ****************\n %f \n*******************************\n",time_cal1);
    // printMatrix_Device(m, m, d_AAT, m, "AAT1");
    float norm_1 = -1;
    if(calculateError){
        norm_1 = 0;
        norm2HH_device(d_AAT, m*m, &norm_1);
    }
    float time2 = timer0.seconds();
    float time_norm = time2-time1;
    // printf("*************norm time1****************\n %f \n*******************************\n",time_norm);

    f2h(d_A,d_Ah,m*n);
    f2h(d_A,d_Ah2,m*n);
    float time3 = timer0.seconds();
    float time_reduce = time3-time2;
    printf("*************Reduce accuracy time****************\n %f \n*******************************\n",time_reduce);
    printMatrix_Device(m,n,d_A,m,"A");
    printMatrix_Device(m, m, d_AAT, m, "AAT");
    // cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
    //                        m, m, n,
    //                        &alpha, d_A, CUDA_R_32F, m,
    //                        d_AT, CUDA_R_32F, n,
    //                        &beta2, d_AAT, CUDA_R_32F, m,
    //                        CUDA_R_32F, algo);
    // assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    for(int i=0;i<calculateTime;i++){
        cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                           m, m, n,
                           &alpha, d_Ah, CUDA_R_16F, m,
                           d_Ah, CUDA_R_16F, m,
                           &beta2, d_AAT, CUDA_R_32F, m,
                           CUDA_R_32F, algo);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    }
    float time4 = (timer0.seconds()-time3)/(dt)calculateTime+time_reduce;
    printMatrix_Device(m, m, d_AAT, m, "AAT");
    printf("*************time2 calculate2****************\n %f \n*******************************\n",time4);
    
    
    float norm_2 = 0;
    if(calculateError){
        printf("norm_1：  %f\n", norm_1);
        norm2HH_device(d_AAT, m*m, &norm_2);
        printf("norm_2：  %f\n", norm_2);
        printf("*************error****************\n %f \n*******************************\n", norm_2/norm_1);
    }

    if(cublasH ) cublasDestroy(cublasH);
    if(h_A     ) cudaFreeHost(h_A); h_A = NULL;
    if(d_A     ) cudaFree(d_A); d_A = NULL;
    if(d_Ah   ) cudaFree(d_Ah); d_Ah = NULL;
    if(d_Ah2   ) cudaFree(d_Ah2); d_Ah2 = NULL;
    if(d_AAT   ) cudaFree(d_AAT); d_AAT = NULL;
    // cudaDeviceReset();
    return 0;
}