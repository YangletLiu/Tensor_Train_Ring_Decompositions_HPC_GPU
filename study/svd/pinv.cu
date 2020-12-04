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


// nvcc -lcublas -lcusolver -lcurand -std=c++11 pinv.cu -o pinv
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


/*
 * Creates the "inverted" sigma matrix starting from the vector of singular values
 *
 */
__global__ void invert_sigma(float * d_S, float * d_Sinv, int n) {
    
    float myeps = 0.001;
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    //# Soft-Thresholding
    
    if (i < n) {
        //### TODO must be done outside
        //### Fill the line with zeros
        for (int j = 0; j < n; j++) {
            d_Sinv[i*n + j] = 0;
        }
        
        if (d_S[i] > d_S[0]*myeps) {
            d_Sinv[i*n + i] = 1/d_S[i];
        } else {
            d_Sinv[i*n + i] = 0;
        }
    }
    __syncthreads();
}

/**
 *
 * d_X : the matrix whose pseudoinverse must be computed
 * n : the number of rows of the matrix
 * p : the number of columns of the matrix
 * d_Xpinv : the pseudoinverse of d_X
 */
int pinv(cublasHandle_t cb_handle, cusolverDnHandle_t cs_handle, float * d_X, int n, int p, float * d_Xpinv, cublasGemmAlgo_t algo) {
    if (n < p) {
        cout << "n must be greater or equal than p; aborting." << endl;
        return -1;
    }
    
    float alpha = 1.0f;
    float beta = 0.0f;
    int np = n*p;
    cudaError_t cudaStat1_pinv = cudaSuccess;
    cudaError_t cudaStat2_pinv = cudaSuccess;
    cudaError_t cudaStat3_pinv = cudaSuccess;
    cudaError_t cudaStat4_pinv = cudaSuccess;
    cudaError_t cudaStat5_pinv = cudaSuccess;
    cublasStatus_t cublas_status_pinv = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status_pinv= CUSOLVER_STATUS_SUCCESS;
    
    dim3 threadsPerBlock(512,1,1);
    dim3 numBlocks((p+512-1)/512,1,1); //for invert_sigma
    
    //### used to control the level of debug output
    int h_lwork = 0, *devInfo = 0;
    float *d_S = 0, *d_U = 0, *d_VH = 0, *d_Sinv = 0, * d_work = 0;
    cudaStat1_pinv = cudaMalloc ((void**)&d_S  , sizeof(float)*p);
    cudaStat2_pinv = cudaMalloc ((void**)&d_U  , sizeof(float)*np);
    cudaStat3_pinv = cudaMalloc ((void**)&d_VH , sizeof(float)*p*p);
    cudaStat4_pinv = cudaMalloc ((void**)&d_Sinv , sizeof(float)*p*p);
    cudaStat5_pinv = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1_pinv);
    assert(cudaSuccess == cudaStat2_pinv);
    assert(cudaSuccess == cudaStat3_pinv);
    assert(cudaSuccess == cudaStat4_pinv);
    assert(cudaSuccess == cudaStat5_pinv);
    
    cusolver_status_pinv = cusolverDnSgesvd_bufferSize(cs_handle, n, p, &h_lwork);
    assert(cusolver_status_pinv == CUSOLVER_STATUS_SUCCESS);

    cudaStat1_pinv = cudaMalloc((void**)&d_work , sizeof(float)*h_lwork);
    assert(cudaSuccess == cudaStat1_pinv);

    //compute the SVD
    cusolver_status_pinv = cusolverDnSgesvd(cs_handle, 'S', 'S', n, p, d_X, n, d_S, d_U, n, d_VH, p, d_work, h_lwork, NULL, devInfo);
    cudaStat1_pinv = cudaDeviceSynchronize();
    assert(cusolver_status_pinv == CUBLAS_STATUS_SUCCESS);
    assert(cudaSuccess == cudaStat1_pinv);
    // printMatrix_Device(n, p, d_U, n, "U");
    // printMatrix_Device(p, 1, d_S, p, "S");
    // printMatrix_Device(p, p, d_VH, p, "VH");

    invert_sigma<<<numBlocks, threadsPerBlock>>>(d_S, d_Sinv, p);
    // printMatrix_Device(p, p, d_Sinv, p, "Sinv");
    //CUBLAS_OP_C 共轭转置操作
    // cublas_status_pinv = cublasSgemm(cb_handle, CUBLAS_OP_T, CUBLAS_OP_T, p, p, p, &alpha, d_VH, p, d_Sinv, p, &beta, d_Sinv, p);
    cublas_status_pinv = cublasGemmEx(cb_handle, CUBLAS_OP_T, CUBLAS_OP_T, p, p, p,
                            &alpha, d_VH, CUDA_R_32F, p, d_Sinv, CUDA_R_32F, p,
                            &beta, d_Sinv, CUDA_R_32F, p,
                            CUDA_R_32F, algo);
    assert(cublas_status_pinv == CUBLAS_STATUS_SUCCESS);
    // printMatrix_Device(p, p, d_Sinv, p, "Sinv");
    // printMatrix_Device(n, p, d_U, n, "U2");
    //d_Xpinv = d_Sinv x d_U^H
    // cublas_status_pinv = cublasSgemm(cb_handle, CUBLAS_OP_N, CUBLAS_OP_T, p, p, n, &alpha, d_Sinv, p, d_U, n, &beta, d_Xpinv, p);
    cublas_status_pinv = cublasGemmEx(cb_handle, CUBLAS_OP_N, CUBLAS_OP_T, p, n, p,
                            &alpha, d_Sinv, CUDA_R_32F, p, d_U, CUDA_R_32F, n,
                            &beta, d_Xpinv, CUDA_R_32F, p,
                            CUDA_R_32F, algo);
    assert(cublas_status_pinv == CUBLAS_STATUS_SUCCESS);
    // printMatrix_Device(p, n, d_Xpinv, p, "Xpinv");
    // cudaStat1_pinv = cudaDeviceSynchronize();
    // assert(cudaSuccess == cudaStat1_pinv);

    cudaFree(d_work);
    cudaFree(devInfo);
    
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VH);
    cudaFree(d_Sinv);
    return 0;
}


int pinv_mbigern_fullRank(cublasHandle_t cb_handle, cusolverDnHandle_t cs_handle, float * d_A, int m, int n, float * d_Apinv, cublasGemmAlgo_t algo) {
    if (m < n) {
        cout << "n must be greater or equal than p; aborting." << endl;
        return -1;
    }
    float alpha = 1.0f;
    float beta = 0.0f;
  
    cudaError_t cudaStat1_pinv = cudaSuccess;
    cudaError_t cudaStat2_pinv = cudaSuccess;
    cudaError_t cudaStat3_pinv = cudaSuccess;
    cudaError_t cudaStat4_pinv = cudaSuccess;
    cudaError_t cudaStat5_pinv = cudaSuccess;
    cublasStatus_t cublas_status_pinv = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status_pinv= CUSOLVER_STATUS_SUCCESS;

    printMatrix_Device(m, n, d_A, m, "A");
    //1. 计算 ATA = AT*A
    float *d_ATA = NULL;
    cudaStat1_pinv = cudaMalloc((void**)&d_ATA, sizeof(float)*n*n);
    assert(cudaSuccess == cudaStat1_pinv);
    cublas_status_pinv = cublasGemmEx(cb_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                            &alpha, d_A, CUDA_R_32F, m, d_A, CUDA_R_32F, m,
                            &beta, d_ATA, CUDA_R_32F, n,
                            CUDA_R_32F, algo);
    assert(cublas_status_pinv == CUBLAS_STATUS_SUCCESS);
    printMatrix_Device(n, n, d_ATA, n, "ATA");
    //2. 计算 ATAinv = inv（ATA）
    
    //3. 计算 Apinv = ATAinv*AT
    return 0;
}


int main(int argc, char*argv[])
{
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

    cublasHandle_t cublasH;
    cublas_status = cublasCreate(&cublasH);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    cusolverDnHandle_t cusolverH = NULL;
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    if(algo == CUBLAS_GEMM_DEFAULT)
        cublasSetMathMode(cublasH, CUBLAS_DEFAULT_MATH);
    else if( algo == CUBLAS_GEMM_DEFAULT_TENSOR_OP)
        cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH);
    
    int n = 5, p = 3;
    float *d_A = NULL, *d_Apinv = NULL;
    cudaMalloc((void**)&d_A, sizeof(float)*n*p);
    cudaMalloc((void**)&d_Apinv, sizeof(float)*n*p);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
    curandGenerateUniform(gen,d_A,n*p);

    // printMatrix_Device(n, p, d_A, n, "A");
    // pinv(cublasH, cusolverH, d_A, n, p, d_Apinv, algo);
    pinv_mbigern_fullRank(cublasH, cusolverH, d_A, n, p, d_Apinv, algo);
    printMatrix_Device(p, n, d_Apinv, p, "Apinv");

    cusolverDnDestroy(cusolverH);
    cublasDestroy(cublasH);

    return 0;
}

//}