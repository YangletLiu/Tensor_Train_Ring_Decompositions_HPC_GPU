#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include "cuda_runtime.h"
#include "curand_kernel.h" 
#include "curand.h"
#include "device_launch_parameters.h" 
#include <assert.h>

#include "cublas_v2.h"
#include <cusolverDn.h>


#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CUSOLVER_CALL(x) do { if((x)!=CUSOLVER_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

//nvcc rSVD2.cu -o rsvd2 -lcurand -lcublas -lcusolver
// /usr/local/cuda/bin/nvcc rSVD2.cu -o rsvd2 -lcurand -lcublas -lcusolver

using namespace std;

typedef float dt;

cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT; //CUBLAS_GEMM_DEFAULT_TENSOR_OP CUBLAS_GEMM_DEFAULT
cublasMath_t mathMode = CUBLAS_DEFAULT_MATH; //CUBLAS_TENSOR_OP_MATH  CUBLAS_DEFAULT_MATH
cudaError_t cudaStat1 = cudaSuccess;
cudaError_t cudaStat2 = cudaSuccess;
cudaError_t cudaStat3 = cudaSuccess;
cudaError_t cudaStat4 = cudaSuccess;
cudaError_t cudaStat5 = cudaSuccess;
bool CalculateError = false;
const float alpha = 1.0, beta = 0.0;
const double tol = 1.e-12;


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

// nvcc -lcublas -lcusolver -lcurand -std=c++11 qr.cu -o qr
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

// void basicSvd(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, float *d_B, const int m, const int n, float *d_UT, float *d_S, float *d_V){
//     float *d_BT = NULL, *d_U = NULL;
//     float *d_work = NULL, *d_rwork = NULL;
//     int *devInfo = NULL;
//     int lwork = 0,  info_gpu = 0;
//     // dt alpha = 1.0;
//     // dt beta = 0.0;
//     cusolverDnHandle_t cusolverH2 = NULL;

//     cusolver_status = cusolverDnCreate(&cusolverH2);
//     assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

//     cudaStat1 = cudaMalloc((void**)&d_BT, sizeof(float)*m*n);
//     cudaStat2 = cudaMalloc((void**)&d_U, sizeof(float)*m*m);
//     cudaStat3 = cudaMalloc ((void**)&devInfo, sizeof(int));
//     assert(cudaStat1 == cudaSuccess);
//     assert(cudaStat2 == cudaSuccess);
//     assert(cudaStat3 == cudaSuccess);

// //转置B
//     cublas_status = cublasSgeam(cublasH,
//                             CUBLAS_OP_T, CUBLAS_OP_N, 
//                             n, m,
//                             &alpha,
//                             d_B, m,
//                             &beta,
//                             d_B, n,
//                             d_BT, n);
//     assert(cublas_status == CUBLAS_STATUS_SUCCESS);
//     cudaStat1 = cudaDeviceSynchronize();
//     assert(cudaStat1 == cudaSuccess);
// // cusolverDnSgesvd_bufferSize 
// // cusolverDnSgesvd_bufferSize

//     cusolver_status = cusolverDnSgesvd_bufferSize(
//         cusolverH2,
//         n,
//         m,
//         &lwork );
//     assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

//     cudaStat1 = cudaMalloc((void**)&d_work , sizeof(float)*lwork);
//     assert(cudaSuccess == cudaStat1);

//     signed char jobu = 'S'; // all m columns of U
//     signed char jobvt = 'S'; // all n columns of VT

//     cusolver_status = cusolverDnSgesvd(
//         cusolverH2,
//         jobu,
//         jobvt,
//         n,
//         m,
//         d_BT,
//         n,
//         d_S,
//         d_V,
//         n,  // ldu
//         d_U,
//         m, // ldvt,
//         d_work,
//         lwork,
//         d_rwork,
//         devInfo);
//     cudaStat1 = cudaDeviceSynchronize();
//     if(cusolver_status == CUSOLVER_STATUS_NOT_INITIALIZED)
//         printf("CUSOLVER_STATUS_NOT_INITIALIZED\n");
//     else if(cusolver_status == CUSOLVER_STATUS_INVALID_VALUE)
//         printf("CUSOLVER_STATUS_INVALID_VALUE\n");
//     else if(cusolver_status == CUSOLVER_STATUS_ARCH_MISMATCH)
//         printf("CUSOLVER_STATUS_ARCH_MISMATCH\n");
//     else if(cusolver_status == CUSOLVER_STATUS_INTERNAL_ERROR)
//         printf("CUSOLVER_STATUS_INTERNAL_ERROR\n");
//     assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
//     assert(cudaSuccess == cudaStat1);

//     cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
//     assert(cudaSuccess == cudaStat1);
//     printf("after gesvd: info_gpu = %d\n", info_gpu);
//     assert(0 == info_gpu);

// //转置U 给UT
//     cublas_status = cublasSgeam(cublasH,
//                             CUBLAS_OP_T, CUBLAS_OP_N, 
//                             m, m,
//                             &alpha,
//                             d_U, m,
//                             &beta,
//                             d_U, m,
//                             d_UT, m);
//     assert(cublas_status == CUBLAS_STATUS_SUCCESS);
//     cudaDeviceSynchronize();

//     if(d_BT) cudaFree(d_BT); d_BT = NULL;
//     if(d_U) cudaFree(d_U); d_U = NULL;
//     if(d_work) cudaFree(d_work); d_work = NULL;
//     if(devInfo) cudaFree(devInfo); devInfo = NULL;
//     if(d_rwork) cudaFree(d_rwork); d_rwork = NULL;
//     if (cusolverH2) cusolverDnDestroy(cusolverH2); cusolverH2 = NULL;
// }


void basicSvd(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, float *d_B, const int m, const int n, float *d_UT, float *d_S, float *d_V){
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

    cusolver_status = cusolverDnDgesvd_bufferSize(
        cusolverH,
        n,
        m,
        &lwork );
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1 = cudaMalloc((void**)&d_work , sizeof(float)*lwork);
    assert(cudaSuccess == cudaStat1);

    signed char jobu = 'S'; // all m columns of U
    signed char jobvt = 'S'; // all n columns of VT

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
    if(cusolver_status == CUSOLVER_STATUS_NOT_INITIALIZED)
        printf("CUSOLVER_STATUS_NOT_INITIALIZED\n");
    else if(cusolver_status == CUSOLVER_STATUS_INVALID_VALUE)
        printf("CUSOLVER_STATUS_INVALID_VALUE\n");
    else if(cusolver_status == CUSOLVER_STATUS_ARCH_MISMATCH)
        printf("CUSOLVER_STATUS_ARCH_MISMATCH\n");
    else if(cusolver_status == CUSOLVER_STATUS_INTERNAL_ERROR)
        printf("CUSOLVER_STATUS_INTERNAL_ERROR\n");
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    printf("after gesvd: info_gpu = %d\n", info_gpu);
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


void basicSvdj(cusolverDnHandle_t cusolverH, float *d_B, int ks, int n, float *d_UT, float *d_S, float *d_V, float tol){
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    
    const int max_sweeps = 100;
    const int econ = 1;
    double residual = 0;
    float *d_work = NULL;
    int *d_info = NULL;
    int executed_sweeps = 0, lwork = 0, info = 0;
    cudaStream_t stream = NULL;
    gesvdjInfo_t gesvdj_params = NULL;
    // printf("tol = %E, default value is machine zero \n", tol);
    // cudaError_t cudaStat1 = cudaSuccess;
    // cudaError_t cudaStat2 = cudaSuccess;
    // cudaError_t cudaStat3 = cudaSuccess;

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    cusolver_status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

/* step 2: configuration of gesvdj */
    cusolver_status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

/* default value of tolerance is machine zero */
    cusolver_status = cusolverDnXgesvdjSetTolerance(gesvdj_params, tol);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

/* default value of max. sweeps is 100 */
    cusolver_status = cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cusolver_status = cusolverDnSgesvdj_bufferSize(
                        cusolverH,
                        jobz,
                        econ,
                        ks,
                        n, 
                        d_B,
                        ks,
                        d_S,
                        d_UT,
                        ks,
                        d_V,
                        n,
                        &lwork,
                        gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cudaStat1 = cudaMalloc((void**)&d_work , sizeof(float)*lwork);
    cudaStat2 = cudaMalloc((void**)&d_info, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    cusolver_status = cusolverDnSgesvdj(
        cusolverH,
        jobz,  /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
               /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ,  /* econ = 1 for economy size */
        ks,     /* nubmer of rows of A, 0 <= m */
        n,     /* number of columns of A, 0 <= n  */
        d_B,   /* m-by-n */
        ks,   /* leading dimension of A */
        d_S,   /* min(m,n)  */
               /* the singular values in descending order */
        d_UT,   /* m-by-m if econ = 0 */
               /* m-by-min(m,n) if econ = 1 */
        ks,   /* leading dimension of U, ldu >= max(1,m) */
        d_V,   /* n-by-n if econ = 0  */
               /* n-by-min(m,n) if econ = 1  */
        n,   /* leading dimension of V, ldv >= max(1,n) */
        d_work,
        lwork,
        d_info,
        gesvdj_params);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    // printMatrix_Device(ks,ks,d_UT,ks,"UT");
    // printMatrix_Device(ks,1,d_S,ks,"S");
    // printMatrix_Device(n,ks,d_V,n,"V");
    cudaStat2 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    cudaStat3 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    if ( 0 == info ){
        printf("gesvdj converges \n");
    }else if ( 0 > info ){
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }else{
        printf("WARNING: info = %d : gesvdj does not converge \n", info );
    }

    cusolver_status = cusolverDnXgesvdjGetSweeps(
        cusolverH,
        gesvdj_params,
        &executed_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cusolver_status = cusolverDnXgesvdjGetResidual(
        cusolverH,
        gesvdj_params,
        &residual);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    printf("residual |A - U*S*V**H|_F = %E \n", residual);
    printf("number of executed sweeps = %d \n", executed_sweeps);

    if(d_work) cudaFree(d_work); d_work = NULL;
    if(d_info) cudaFree(d_info); d_info = NULL;
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
    printf("%d\n", lwork);
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

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

    printf("after geqrf: info_gpu = %d\n", info_gpu);
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

    printf("after orgqr: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

    if (d_work) cudaFree(d_work); d_work = NULL;
    if (devInfo) cudaFree(devInfo); devInfo = NULL;
    if (d_tau) cudaFree(d_tau); d_tau = NULL;
}

void basicRandSVD(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, float *d_A, 
    const int m, const int n, const int k, const int s,const int p, cublasGemmAlgo_t algo,
    float *d_U, float *d_S, float *d_V){

    GPUTimer timer1;


    dt alpha = 1.0;
    dt beta = 0.0;

    const int ks = k+s;
    float time = 0;

    printf("\n---------------------------------\n random B \n");
///获得随机正太分布矩阵B
    timer1.start();

    curandGenerator_t gen;
    float *d_B, *d_C, *d_UT;
    cudaStat1 = cudaMalloc((void**)&d_B, sizeof(float)*n*ks); //d_V
    cudaStat2 = cudaMalloc((void**)&d_C, sizeof(float)*m*ks); //d_U
    cudaStat3 = cudaMalloc((void**)&d_UT, sizeof(float)*ks*ks);
    
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormal(gen, d_B, n*ks, 0, 1);
    
    // printMatrix_Device(n,ks,d_B,n,"B");
    // printMatrix_Device(m,n,d_A,m,"A");
///矩阵A 乘以B 获得矩阵C
    cublas_status = cublasGemmEx(cublasH,
                           CUBLAS_OP_N, 
                           CUBLAS_OP_N,
                           m, 
                           ks, 
                           n,
                           &alpha,
                           d_A, 
                           CUDA_R_32F, 
                           m,
                           d_B, 
                           CUDA_R_32F, 
                           n,
                           &beta,
                           d_C, 
                           CUDA_R_32F, 
                           m,
                           CUDA_R_32F, 
                           algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);

///qr(C) 获得Q矩阵
    // printMatrix_Device(m,ks,d_C,m, "C1");
    
    basicQR(cusolverH, d_C, m, ks);

    for(int i = 0; i < p; ++i){
    // B = A'*Q (d_C)
        cublas_status = cublasGemmEx(cublasH,
                           CUBLAS_OP_T, 
                           CUBLAS_OP_N,
                           n, 
                           ks, 
                           m,
                           &alpha,
                           d_A, 
                           CUDA_R_32F, 
                           m, 
                           d_C, 
                           CUDA_R_32F, 
                           m,
                           &beta,
                           d_B, 
                           CUDA_R_32F, 
                           n,
                           CUDA_R_32F, 
                           algo);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    // Q(d_B) = QR(B)
        basicQR(cusolverH, d_B, n, ks);

    // C = A * Q (d_B)
        cublas_status = cublasGemmEx(cublasH,
                           CUBLAS_OP_N, 
                           CUBLAS_OP_N,
                           m, 
                           ks, 
                           n,
                           &alpha,
                           d_A, 
                           CUDA_R_32F, 
                           m, 
                           d_B, 
                           CUDA_R_32F, 
                           n,
                           &beta,
                           d_C, 
                           CUDA_R_32F, 
                           m,
                           CUDA_R_32F, 
                           algo);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    // Q(d_C) = QR(C)
        basicQR(cusolverH, d_C, m, ks);
    }

    // printMatrix_Device(m,ks,d_C,m, "Q");
    // printMatrix_Device(m,n,d_A,m, "A");

///d_B = d_C^T(Q^T)*d_A
    cublas_status = cublasGemmEx(cublasH,
                           CUBLAS_OP_T, 
                           CUBLAS_OP_N,
                           ks, 
                           n, 
                           m,
                           &alpha,
                           d_C, 
                           CUDA_R_32F, 
                           m,
                           d_A, 
                           CUDA_R_32F, 
                           m,
                           &beta,
                           d_B, 
                           CUDA_R_32F, 
                           ks,
                           CUDA_R_32F, 
                           algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // printMatrix_Device(ks,n,d_B,ks, "B");
    printf("ks:%d n:%d for B\n", ks, n);

///d_UT d_S d_V = svdj(d_B,'econ')
    // basicSvdj(cusolverH, d_B, ks, n, d_UT, d_S, d_V, tol);
    basicSvd(cusolverH, cublasH, d_B, ks, n, d_UT, d_S, d_V);
/// U = Q*UT
    cublas_status = cublasGemmEx(cublasH,
                           CUBLAS_OP_N, 
                           CUBLAS_OP_N,
                           m, 
                           ks, 
                           ks,
                           &alpha,
                           d_C, 
                           CUDA_R_32F, 
                           m,
                           d_UT, 
                           CUDA_R_32F, 
                           ks,
                           &beta,
                           d_U, 
                           CUDA_R_32F, 
                           m,
                           CUDA_R_32F, 
                           algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    // printMatrix_Device(m,ks,d_U,ks,"U");
    time = timer1.seconds();
    printf("random b time:%f\n",time);
    // printMatrix_Device(m,ks,d_C,m, "C");
    if(d_B) cudaFree(d_B); d_B = NULL;
    if(d_C) cudaFree(d_C); d_C = NULL;
    if(d_UT) cudaFree(d_UT); d_UT = NULL;
}


int main(int argc, char*argv[])
{   
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    int q = 500, r = 50; 
    int d1 = q, d2 = q, d3 = q;
    const int m = d1;
    const int n = d2*d3;
    const long long mn = (long long)m*(long long)n;
    const int k = min(m,n)/2;
    const int s = 8;
    const int p = 2;
    const int step = 10;
    const float eps = 1.e-5;
/*       | 1 2 3 |
 *   A = | 4 5 6 |
 *       | 2 1 1 |
 */

    //float h_A[m*n] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0}; 
    float *h_A, *d_A = NULL, *d_U = NULL, *d_S = NULL, *d_V = NULL; // linear memory of GPU  
    cudaStat1 = cudaMallocHost((void**)&h_A, sizeof(float)*mn);
    assert(cudaSuccess == cudaStat1);
    genTTTensor(h_A,d1,d2,d3,r);
  

    GPUTimer timer1;
    timer1.start();
    float time = 0;
// step 1: create cusolver/cublas handle

    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
      
    cublas_status = cublasSetMathMode(cublasH, mathMode);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

// step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(float) * mn);
    assert(cudaSuccess == cudaStat1);
   
    float before_ANorm2 = 0;
    cudaStat1 = cudaMemcpy(d_A, h_A, sizeof(float) * mn, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    cudaStat2 = cudaMalloc ((void**)&d_U  , sizeof(float) * m * (k+s));
    cudaStat3 = cudaMalloc ((void**)&d_S  , sizeof(float) * (k+s));
    cudaStat4 = cudaMalloc ((void**)&d_V  , sizeof(float) * n * (k+s));

    basicRandSVD(cusolverH, cublasH, d_A, m, n, k, s, p, algo, d_U, d_S, d_V);
        
    time = timer1.seconds();
    printf("\n********************************\n random b time:%f\n**********************************\n",time);
    
// free resources
    if (d_A    ) cudaFree(d_A);
    if (d_U    ) cudaFree(d_U);
    if (d_S    ) cudaFree(d_S);
    if (d_V    ) cudaFree(d_V);

    if (cublasH ) cublasDestroy(cublasH);   
    if (cusolverH) cusolverDnDestroy(cusolverH);   


  return 0;
}