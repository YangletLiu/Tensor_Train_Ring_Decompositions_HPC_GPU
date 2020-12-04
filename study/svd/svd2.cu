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
cudaError_t cudaStat1 = cudaSuccess;
cudaError_t cudaStat2 = cudaSuccess;
cudaError_t cudaStat3 = cudaSuccess;
cudaError_t cudaStat4 = cudaSuccess;
cudaError_t cudaStat5 = cudaSuccess;
cudaError_t cudaStat6 = cudaSuccess;
const float alpha = 1.0, beta = 0.0;

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

// /usr/local/cuda/bin/nvcc -lcublas -lcusolver -lcurand -std=c++11 svd2.cu -o svd2
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


void basicSvd(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, float *d_B, const int m, const int n, float *d_UT, float *d_S, float *d_V){
    printf("start svd for m: %d, n:%d\n", m, n);
    float *d_BT = NULL, *d_U = NULL;
    float *d_work = NULL, *d_rwork = NULL;
    int *devInfo = NULL;
    int lwork = 0,  info_gpu = 0;

    cudaStat1 = cudaMalloc((void**)&d_BT, sizeof(float)*m*n);
    cudaStat2 = cudaMalloc((void**)&d_U, sizeof(float)*m*m);
    cudaStat3 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);

//转置B
    cublas_status = cublasSgeam(cublasH,
                            CUBLAS_OP_T, CUBLAS_OP_N, 
                            n, m,
                            &alpha,
                            d_B, m,
                            &beta,
                            d_B, n,
                            d_BT, n);
    assert(cublas_status == CUSOLVER_STATUS_SUCCESS);
    cudaDeviceSynchronize();
    cout <<"m: "<< m <<"  n: "<< n <<endl;
    cusolver_status = cusolverDnDgesvd_bufferSize(
        cusolverH,
        n,
        m,
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1 = cudaMalloc((void**)&d_work , sizeof(float)*lwork);
    cout << "svd sapce: " << lwork << "GB"<<endl;
    assert(cudaSuccess == cudaStat1);

    signed char jobu = 'S'; // all m columns of U
    signed char jobvt = 'S'; // all n columns of VT
    cout << "n: " << n <<" m:" << m <<endl;
    cusolver_status = cusolverDnSgesvd(
        cusolverH,
        jobu,
        jobvt,
        n,
        m,
        d_BT,
        n,
        d_S,
        d_V,
        n,  // ldu
        d_U,
        m, // ldvt,
        d_work,
        lwork,
        d_rwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    if(CUSOLVER_STATUS_INVALID_VALUE == cusolver_status)
        cout << "CUSOLVER_STATUS_INVALID_VALUE" <<endl;
    if(CUSOLVER_STATUS_ARCH_MISMATCH == cusolver_status)
        cout << "CUSOLVER_STATUS_ARCH_MISMATCH" <<endl;
    if(CUSOLVER_STATUS_INTERNAL_ERROR == cusolver_status)
        cout << "CUSOLVER_STATUS_INTERNAL_ERROR" <<endl;
    if(CUSOLVER_STATUS_NOT_INITIALIZED == cusolver_status)
        cout << "CUSOLVER_STATUS_NOT_INITIALIZED" <<endl;
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    // printf("after gesvd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

//转置U 给UT
    cublas_status = cublasSgeam(cublasH,
                            CUBLAS_OP_T, CUBLAS_OP_N, 
                            m, m,
                            &alpha,
                            d_U, m,
                            &beta,
                            d_U, m,
                            d_UT, m);
    assert(cublas_status == CUSOLVER_STATUS_SUCCESS);
    cudaDeviceSynchronize();

    if(d_BT) cudaFree(d_BT); d_BT = NULL;
    if(d_U) cudaFree(d_U); d_U = NULL;
    if(d_work) cudaFree(d_work); d_work = NULL;
    if(devInfo) cudaFree(devInfo); devInfo = NULL;
    if(d_rwork) cudaFree(d_rwork); d_rwork = NULL;
}




int main(int argc, char*argv[])
{
	cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    
    int k = 2360;
    const int m = k/10/8*8+8;
    const long int n = k*k;
    
/*       | 1 2  |
 *   A = | 4 5  |
 *       | 2 1  |
 */
    float *A = NULL;
    srand(time(NULL));
    //float A[lda*n] = { 0.4, 0.2, 0.1, 0.5, 0.3, 0.2};
    cudaHostAlloc((void**)&A, sizeof(float)*m*n,0);
    for(int i = 0; i < m*n; ++i){
        A[i] = (float) rand()*1.0 / RAND_MAX*1.0;
    }
    // float U[lda*m]; // m-by-m unitary matrix 
    // float VT[lda*n];  // n-by-n unitary matrix
    // float S[n]; // singular value
    // float S_exact[n] = {7.065283497082729, 1.040081297712078};

    float *d_A = NULL;
    float *d_S = NULL;
    float *d_U = NULL;
    float *d_VT = NULL;

    // printf("A = (matlab base-1)\n");
    // printMatrix(m, n, A, lda, "A");
    // printf("=====\n");

// step 1: create cusolverDn/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

// step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(float)*m*n);
    cudaStat2 = cudaMalloc ((void**)&d_S  , sizeof(float)*m);
    cudaStat3 = cudaMalloc ((void**)&d_U  , sizeof(float)*m*m);
    cudaStat4 = cudaMalloc ((void**)&d_VT , sizeof(float)*m*n); 
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);

    basicSvd(cusolverH, cublasH, d_A, m, n, d_U, d_S, d_VT);

// free resources
    if (d_A    ) cudaFree(d_A);
    if (d_S    ) cudaFree(d_S);
    if (d_U    ) cudaFree(d_U);
    if (d_VT   ) cudaFree(d_VT);

    if (cublasH ) cublasDestroy(cublasH);
    if (cusolverH) cusolverDnDestroy(cusolverH);

    cudaDeviceReset();

    return 0;
}
//}