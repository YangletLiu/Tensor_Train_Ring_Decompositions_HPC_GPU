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


// nvcc -lcublas -lcusolver -lcurand -std=c++11 svdj.cu -o svdj
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

int main(int argc, char*argv[])
{
	cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    gesvdjInfo_t gesvdj_params = NULL;
    
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;


    dt *d_A = NULL;  /* device copy of A */
    dt *d_S = NULL;  /* singular values */
    dt *d_U = NULL;  /* left singular vectors */
    dt *d_V = NULL;  /* right singular vectors */
    //dt *h_S = NULL;
    int *d_info = NULL;  /* error info */
    int lwork = 0;       /* size of workspace */
    dt *d_work = NULL; /* devie workspace for gesvdj */
    int info = 0; 
    // const int m = h_n_array[work_time];
    // const int n = h_n_array[1] * h_n_array[2] * h_n_array[3];
    int q = 1200, r = 50; 
    int d1 = q, d2 = q, d3 = q;
    const int m = d1;
    const int n = d2*d3;
    const int econ_s = min(m,n); //1500
    const int lda = m;  // 1500
    const int ldu = lda;  // 1500
    const int ldv = n;  // 2250000

    /* configuration of gesvdj  */
    const double tol = 1.e-8;
    const int max_sweeps = 100;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    const int econ = 1;
   
    /* numerical results of gesvdj  */
    double residual = 0;
    int executed_sweeps = 0;


    //cout<<"size:"<<i<<endl;
    srand(time(NULL));
    float *A = NULL;  //  b*a
    cudaHostAlloc((void**)&A,sizeof(float)*m*n,0);
    
    // for(long i = 0; i < m*n; ++i) {
    //     A[i]= (dt) (rand()*1.0/RAND_MAX*1.0);
    // }
    genTTTensor(A,d1,d2,d3,r);
    // printf("step1 ------------ \n");
    // printf("tol = %E, default value is machine zero \n", tol);
    // printf("max. sweeps = %d, default value is 100\n", max_sweeps);
    // printf("econ = %d \n", econ);
    GPUTimer timer1;
    timer1.start();
    /* step 1: create cusolver handle, bind a stream */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 2: configuration of gesvdj */
    status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of tolerance is machine zero */
    status = cusolverDnXgesvdjSetTolerance(
        gesvdj_params,
        tol);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of max. sweeps is 100 */
    status = cusolverDnXgesvdjSetMaxSweeps(
        gesvdj_params,
        max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 3: copy A and B to device */
    cudaStat1 = cudaMalloc ((void**)&d_A   , sizeof(dt)*lda*n);
    cudaStat2 = cudaMalloc ((void**)&d_S   , sizeof(dt)*econ_s);
    cudaStat3 = cudaMalloc ((void**)&d_U   , sizeof(dt)*ldu*econ_s);
    cudaStat4 = cudaMalloc ((void**)&d_V   , sizeof(dt)*ldv*econ_s);//
    cudaStat5 = cudaMalloc ((void**)&d_info, sizeof(int));
    //cudaHostAlloc((void**) &h_S, sizeof(dt)*econ_s,0);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(dt)*lda*n, cudaMemcpyHostToDevice);
    //cudaStat1 = cudaMemcpyAsync(d_A, h_tensor, sizeof(dt)*lda*n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    cout<< " m:" << m << " n:" << n  <<endl;
    //cout << " lda:" << lda << " ldu:" << ldu << " ldv:" << ldv << " eps:" << eps <<endl;

/* step 4: query workspace of SVD */
    status = cusolverDnSgesvdj_bufferSize(
        cusolverH,
        jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
              /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ, /* econ = 1 for economy size */
        m,    /* nubmer of rows of A, 0 <= m */
    // const int econ_s = min(m,n); //1500
    // const int lda = m;  // 1500
    // const int ldu = lda;  // 1500
    // const int ldv = n;  // 2250000
        n,    /* number of columns of A, 0 <= n  */
        d_A,  /* m-by-n */
        lda,  /* leading dimension of A */
        d_S,  /* min(m,n) */
              /* the singular values in descending order */
        d_U,  /* m-by-m if econ = 0 */
              /* m-by-min(m,n) if econ = 1 */
        ldu,  /* leading dimension of U, ldu >= max(1,m) */
        d_V,  /* n-by-n if econ = 0  */
              /* n-by-min(m,n) if econ = 1  */
        ldv,  /* leading dimension of V, ldv >= max(1,n) */
        &lwork,
        gesvdj_params);
    if( CUSOLVER_STATUS_NOT_INITIALIZED == status)
        cout << "CUSOLVER_STATUS_NOT_INITIALIZED" <<endl;

    if( CUSOLVER_STATUS_INVALID_VALUE == status)
        cout << "CUSOLVER_STATUS_INVALID_VALUE  " <<endl;

    if( CUSOLVER_STATUS_INTERNAL_ERROR == status)
        cout << "CUSOLVER_STATUS_INTERNAL_ERROR" <<endl; 

    assert(CUSOLVER_STATUS_SUCCESS == status);
    cout << "-------------------------\npart one work space: " <<sizeof(dt)*lwork << endl;
    cudaStat1 = cudaMalloc((void**)&d_work , sizeof(dt)*lwork);
    assert(cudaSuccess == cudaStat1);

    // printMatrix_Device(lda,n,d_A,lda,"A");
    // printMatrix(lda,n,h_tensor,lda,"hA");

    dt time = 0.0f;
    GPUTimer timer;
    timer.start();
    /* step 5: compute SVD */
    status = cusolverDnSgesvdj(
        cusolverH,
        jobz,  /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
               /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ,  /* econ = 1 for economy size */
        m,     /* nubmer of rows of A, 0 <= m */
        n,     /* number of columns of A, 0 <= n  */
        d_A,   /* m-by-n */
        lda,   /* leading dimension of A */
        d_S,   /* min(m,n)  */
               /* the singular values in descending order */
        d_U,   /* m-by-m if econ = 0 */
               /* m-by-min(m,n) if econ = 1 */
        ldu,   /* leading dimension of U, ldu >= max(1,m) */
        d_V,   /* n-by-n if econ = 0  */
               /* n-by-min(m,n) if econ = 1  */
        ldv,   /* leading dimension of V, ldv >= max(1,n) */
        d_work,
        lwork,
        d_info,
        gesvdj_params);
    cudaStat1 = cudaDeviceSynchronize();
    time = timer.seconds();
    cout << "----------------------------------\n part one svd run time: " << time << endl;

    // printMatrix_Device(ldu,econ_s,d_U,ldu,"U");
    // printMatrix_Device(500,1,d_U,500,"U");
    // printMatrix_Device(econ_s,1,d_S,econ_s,"S");
    // printMatrix_Device(ldv,econ_s,d_V,ldv,"V");

    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    //cudaStat1 = cudaMemcpy(h_S, d_S, sizeof(dt)*econ_s, cudaMemcpyDeviceToHost);  
    cudaStat2 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    cudaStat3 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
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
    

    status = cusolverDnXgesvdjGetSweeps(
        cusolverH,
        gesvdj_params,
        &executed_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    status = cusolverDnXgesvdjGetResidual(
        cusolverH,
        gesvdj_params,
        &residual);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    printf("residual |A - U*S*V**H|_F = %E \n", residual );
    printf("number of executed sweeps = %d \n", executed_sweeps );

    time = timer1.seconds();
    printf("random b time:%f\n",time);
/*  free resources  */
    if (d_A    ) cudaFree(d_A);
    if (d_S    ) cudaFree(d_S);
    if (d_U    ) cudaFree(d_U);
    if (d_V    ) cudaFree(d_V);
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);
    if (A      ) cudaFreeHost(A);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);
    if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);

    return 0;
}

//}