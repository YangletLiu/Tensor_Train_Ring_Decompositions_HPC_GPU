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

// nvcc -lcublas -lcusolver -lcurand -std=c++11 svd.cu -o svd
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

int main(int argc, char*argv[])
{
	//for(int i=100;i<=1000;i=i+100){
    cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH);
    int m = 10000;//
    int n = 100;//  m必须>=n
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    //cout<<"size:"<<i<<endl;
    srand(time(NULL));
    float *A = NULL;  //  b*a
    cudaHostAlloc((void**)&A,sizeof(float)*m*n,0);
    
    for(long i = 0; i < m*n; ++i) {
    	A[i]= (dt) (rand()*1.0 / RAND_MAX *1.0);
    }
    // float A[6] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
    float *d_A;
    cudaMalloc((void**)&d_A,sizeof(float)*m*n);
    cudaStat1 = cudaMemcpyAsync(d_A, A,sizeof(float)*m*n,cudaMemcpyHostToDevice,0);
    assert(cudaStat1 == cudaSuccess);
    cudaDeviceSynchronize();

    float *d_S = NULL;
    float *d_U = NULL;
    float *d_VT = NULL;
    int *devInfo = NULL;
    float *d_work = NULL;
    float *d_rwork = NULL;
    //float *d_W = NULL; 
    int lwork = 0;
    cudaStat1 = cudaMalloc ((void**)&d_S  , sizeof(double)*n);
    cudaStat2 = cudaMalloc ((void**)&d_U  , sizeof(double)*m*n);
    cudaStat3 = cudaMalloc ((void**)&d_VT , sizeof(double)*n*n);
    cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);

    cout<< " m:" << m << " n:" << n  <<endl;
    cusolver_status=  cusolverDnDgesvd_bufferSize(
        cusolverH,
        m,
        n,
        &lwork );
    if(CUSOLVER_STATUS_INVALID_VALUE == cusolver_status)
        cout << "CUSOLVER_STATUS_INVALID_VALUE" <<endl;
    if(CUSOLVER_STATUS_ARCH_MISMATCH == cusolver_status)
        cout << "CUSOLVER_STATUS_ARCH_MISMATCH" <<endl;
    if(CUSOLVER_STATUS_INTERNAL_ERROR == cusolver_status)
        cout << "CUSOLVER_STATUS_INTERNAL_ERROR" <<endl;
    if(CUSOLVER_STATUS_NOT_INITIALIZED == cusolver_status)
        cout << "CUSOLVER_STATUS_NOT_INITIALIZED" <<endl;
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cudaStat1 = cudaMalloc((void**)&d_work , sizeof(double)*lwork);
    assert(cudaStat1 == cudaSuccess);
    signed char jobu = 'S'; 
    signed char jobvt = 'S';

    dt time = 0.0f;
    GPUTimer timer;
    timer.start();
    cusolver_status = cusolverDnSgesvd (
        cusolverH,
        jobu,
        jobvt,
        m,
        n,
        d_A,
        m,
        d_S,
        d_U,
        m,  // ldu
        d_VT,
        n, // ldvt,
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
    time = timer.seconds();
    cout << "----------------------------------\npart one svd run time: " << time << "s"<< endl;
    printMatrix_Device(n,1,d_S,n,"S");

    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(devInfo);
    cudaFree(d_work);
    cudaFree(d_A);
    cudaFreeHost(A);
   	//cudaFree(d_rwork);
	cusolverDnDestroy(cusolverH);
	cudaDeviceSynchronize();

}
//}