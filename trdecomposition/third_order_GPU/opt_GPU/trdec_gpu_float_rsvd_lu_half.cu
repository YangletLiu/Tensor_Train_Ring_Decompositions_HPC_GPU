#include "head.h"

cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP; //CUBLAS_GEMM_DEFAULT_TENSOR_OP CUBLAS_GEMM_DEFAULT
cublasMath_t mathMode = CUBLAS_TENSOR_OP_MATH ; //CUBLAS_TENSOR_OP_MATH  CUBLAS_DEFAULT_MATH
cudaError_t cudaStat1 = cudaSuccess;
cudaError_t cudaStat2 = cudaSuccess;
cudaError_t cudaStat3 = cudaSuccess;
cudaError_t cudaStat4 = cudaSuccess;
cudaError_t cudaStat5 = cudaSuccess;
cudaError_t cudaStat6 = cudaSuccess;
cudaError_t cudaStat7 = cudaSuccess;
bool CalculateError = false;
const float alpha = 1.0, beta = 0.0;
const double tol = 1.e-12;

double norm2HH(float *A, long long len){
  // printf("%lld\n", len);
  double norm2Result = 0.0;
  for(long long i = 0; i < len; ++i){
    norm2Result += (A[i] * A[i]);
  }
  norm2Result = sqrtf(norm2Result);
  return norm2Result;
}

void basicEig(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, dt *d_A, const int m, dt *d_W){  
    int *devInfo = NULL;
    dt *d_work = NULL;
    int  lwork = 0;
    int info_gpu = 0;
    printf("eign %d\n", m);
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

void basicQR(cusolverDnHandle_t cusolverH, float *d_A,int m,int n)
{
    float *d_work = NULL, *d_tau = NULL;
    int *devInfo = NULL;
     int lwork_geqrf = 0;
    int lwork_orgqr = 0;
    int lwork = 0;
    int info_gpu = 0;
    cudaMalloc((void**)&d_tau, sizeof(float)*n);
    cudaMalloc ((void**)&devInfo, sizeof(int));

    cusolverDnSgeqrf_bufferSize(
        cusolverH,
        m,
        n,
        d_A,
        m,
        &lwork_geqrf);
    cusolverDnSorgqr_bufferSize(
        cusolverH,
        m,
        n,
        n,
        d_A,
        m,
        d_tau,
        &lwork_orgqr);
    lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
    cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    cusolverDnSgeqrf(
        cusolverH,
        m,
        n,
        d_A,
        m,
        d_tau,
        d_work,
        lwork,
        devInfo);
    cusolverDnSorgqr(
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

    cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
     //printf("after geqrf: info_gpu = %d\n", info_gpu);

    if (d_work) cudaFree(d_work); d_work = NULL;
    if (devInfo) cudaFree(devInfo); devInfo = NULL;
    if (d_tau) cudaFree(d_tau); d_tau = NULL;
}

void basicSvdj(cusolverDnHandle_t cusolverH, float *d_B, int ks, int n, float *d_UT, float *d_S, float *d_V){
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

void basicSvd(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, float *d_B, const int m, const int n, float *d_UT, float *d_S, float *d_V){
    // printf("start svd for m: %d, n:%d\n", m,n);
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
    // cout << "n: " << n <<" m:" << m <<endl;
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

void basicRandSVD(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, float *d_A, 
    const int m, const int n, const int k, const int s,const int p, cublasGemmAlgo_t algo,
    float *d_U, float *d_S, float *d_V){

    GPUTimer timer1;
    
    const int ks = k+s;
    float time = 0;

    // printf("\n---------------------------------\n random B \n---------------------------------\n");
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
    // printf("ks:%d n:%d for B\n", ks, n);

///d_UT d_S d_V = svdj(d_B,'econ')
    // basicSvdj(cusolverH, d_B, ks, n, d_UT, d_S, d_V);
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
    // printf("\n---------------------------------\nrandom b time:%f\n---------------------------------\n",time);
    // printMatrix_Device(m,ks,d_C,m, "C");
    if(d_B) cudaFree(d_B); d_B = NULL;
    if(d_C) cudaFree(d_C); d_C = NULL;
    if(d_UT) cudaFree(d_UT); d_UT = NULL;
}

void basicRandSVD_LU(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, float *d_A, 
    const int m, const int n, const int k, const int s,const int p, cublasGemmAlgo_t algo,
    float *d_U, float *d_SVT){
    int ks = k + s;
    float time = 0;

    // printf("\n---------------------------------\n random B \n---------------------------------\n");
///获得随机正太分布矩阵B
    // timer1.start();
    cout << "m: " <<m<<"  n: "<<n <<  "  ks: "<<ks<<endl;
    curandGenerator_t gen;
    float *d_B, *d_C, *d_UT, *d_W;
    half *d_Ah, *d_Bh, *d_Ch, *d_UTh;
    cudaStat1 = cudaMalloc((void**)&d_B, sizeof(float)*n*ks); //d_V
    cudaStat2 = cudaMalloc((void**)&d_C, sizeof(float)*m*ks); //d_U
    cudaStat3 = cudaMalloc((void**)&d_UT, sizeof(float)*ks*ks);
    cudaStat4 = cudaMalloc((void**)&d_Ah, sizeof(half)*m*n);
    cudaStat5 = cudaMalloc((void**)&d_Bh, sizeof(half)*n*ks);
    cudaStat6 = cudaMalloc((void**)&d_Ch, sizeof(half)*m*ks);
    cudaStat7 = cudaMalloc((void**)&d_UTh, sizeof(half)*ks*ks);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);
    assert(cudaStat5 == cudaSuccess);
    assert(cudaStat6 == cudaSuccess);
    assert(cudaStat7 == cudaSuccess);

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormal(gen, d_B, n*ks, 0, 1); 
    
    // GPUTimer timer4;
    // timer4.start();
    f2h(d_A,d_Ah,(long)m*n);
    f2h(d_B,d_Bh,(long)n*ks);
    // cout<<"数据A和B 降精度时间：  "<<timer4.seconds()<<endl;

    assert(cudaStat1 == cudaSuccess);
    // printMatrix_Device(n,ks,d_B,n,"B");
    // printMatrix_Device(m,n,d_A,m,"A");
///矩阵A 乘以B 获得矩阵C
    // printMatrix_Device(10, 10, d_A, 10, "A1");
    // printMatrix_Device(10, 10, d_B, 10, "B1");
    cublas_status = cublasGemmEx(cublasH,
                           CUBLAS_OP_N, 
                           CUBLAS_OP_N,
                           m,
                           ks,
                           n,
                           &alpha,
                           d_Ah, 
                           CUDA_R_16F, 
                           m,
                           d_Bh, 
                           CUDA_R_16F, 
                           n,
                           &beta,
                           d_C, 
                           CUDA_R_32F, 
                           m,
                           CUDA_R_32F, 
                           algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // printMatrix_Device(10, 10, d_C, 10, "C1");
///qr(C) 获得Q矩阵
    // printMatrix_Device(m,ks,d_C,m, "C1");
    
    basicQR(cusolverH, d_C, m, ks);
    f2h(d_C,d_Ch,m*ks);

    for(int i = 0; i < p; ++i){
    // B = A'*Q (d_C)
        
        cublas_status = cublasGemmEx(cublasH,
                           CUBLAS_OP_T, 
                           CUBLAS_OP_N,
                           n,
                           ks,
                           m,
                           &alpha,
                           d_Ah,
                           CUDA_R_16F,
                           m,
                           d_Ch,
                           CUDA_R_16F,
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
        f2h(d_B,d_Bh,n*ks);

    // C = A * Q (d_B)
        cublas_status = cublasGemmEx(cublasH,
                           CUBLAS_OP_N, 
                           CUBLAS_OP_N,
                           m, 
                           ks, 
                           n,
                           &alpha,
                           d_Ah, 
                           CUDA_R_16F, 
                           m, 
                           d_Bh, 
                           CUDA_R_16F, 
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
        f2h(d_C,d_Ch,m*ks);
        cudaStat1 = cudaDeviceSynchronize();
        assert(cudaStat1 == cudaSuccess);
    }
    if(d_Bh) cudaFree(d_Bh);d_Bh=NULL;

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
                           d_Ch, 
                           CUDA_R_16F, 
                           m,
                           d_Ah, 
                           CUDA_R_16F, 
                           m,
                           &beta,
                           d_B, 
                           CUDA_R_32F, 
                           ks,
                           CUDA_R_32F, 
                           algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    if(d_Ah) cudaFree(d_Ah);d_Ah=NULL;
    // printMatrix_Device(ks,n,d_B,10, "B");
    // printf("ks:%d n:%d for B\n", ks, n);
    cudaMalloc ((void**)&d_W, sizeof(dt) * ks);
///对B进行分解
    // GPUTimer timerSVD;
    // timerSVD.start();
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, ks, ks, n, &alpha, d_B, ks, d_B, ks, &beta, d_U, ks);
    // printMatrix_Device(ks, ks, d_U, 10, "BBT");
    // basicSvd(cusolverH, cublasH, d_B, ks, n, d_UT, d_S, d_V);
    // printMatrix_Device(10, 10, d_U, 10, "U");
    basicEig(cusolverH, cublasH, d_U, ks, d_W);
    // printMatrix_Device(10, 10, d_U, 10, "U1");
    // printMatrix_Device(ks, ks, d_U, 10, "U");
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    if (d_W) cudaFree(d_W);
    dim3 threads(512,1,1);
    dim3 blocksUinv((ks*ks+512-1)/512,1,1);
    matrixInvertColumn<<<blocksUinv, threads>>>(d_U, d_UT, ks, ks);

    // printMatrix_Device(10, 10, d_UT, 10, "U2");
    // cudaStat1 = cudaDeviceSynchronize();
    // assert(cudaSuccess == cudaStat1);
    
    // printMatrix_Device(ks, ks, d_UT, 10, "UT");

    cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                           ks, n, ks,
                           &alpha, d_UT, CUDA_R_32F, ks,
                           d_B, CUDA_R_32F, ks,
                           &beta, d_SVT, CUDA_R_32F, ks,
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    // printMatrix_Device(10, 10, d_SVT, 10, "SVT");
    // cudaStat1 = cudaDeviceSynchronize();
    // assert(cudaSuccess == cudaStat1);
    f2h(d_UT,d_UTh,ks*ks);
/// U = Q*UT
    cublas_status = cublasGemmEx(cublasH,
                           CUBLAS_OP_N, 
                           CUBLAS_OP_N,
                           m,
                           ks,
                           ks,
                           &alpha,
                           d_Ch, 
                           CUDA_R_16F, 
                           m,
                           d_UTh,
                           CUDA_R_16F, 
                           ks,
                           &beta,
                           d_U, 
                           CUDA_R_32F, 
                           m,
                           CUDA_R_32F, 
                           algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    // printMatrix_Device(10, 10, d_U, 10, "U2");
    // printf("\n***********************\n time svd: %f \n*********************\n",  timerSVD.seconds());
    // printMatrix_Device(m,ks,d_U,ks,"U");
    // time = timer1.seconds();
    // printf("\n---------------------------------\nrandom b time:%f\n---------------------------------\n",time);
    // printMatrix_Device(m,ks,d_C,m, "C");
    if(d_B) cudaFree(d_B); d_B = NULL;
    if(d_C) cudaFree(d_C); d_C = NULL;
    if(d_Ch) cudaFree(d_Ch); d_Ch = NULL;
    if(d_UT) cudaFree(d_UT); d_UT = NULL;
    if(d_UTh) cudaFree(d_UTh); d_UTh = NULL;
}


float trdec_gpu_float(float* h_tensor, int* h_n_array, int* h_r_array,double *eps, float* singleError, float* singleComRatio ){
    // cout << "algo: " << algo <<endl;
    GPUTimer timer0;
    timer0.start();

    // GPUTimer timer1;
    // timer1.start();

    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;

    dt *d_A = NULL;  /* device copy of A */
    dt *d_U = NULL;  /* left singular vectors */
    dt *d_G1 = NULL;

    dt *h_G1 = NULL;

    const int m = h_n_array[0];  //1500
    const int n = h_n_array[1] * h_n_array[2];  //1500 * 1500 = 2250000
    int k = h_r_array[0] * h_r_array[1];
    const int s = 0;
    const int p = 1;
    // int ldu = m;
    int ks = k+s;
//For test
    // int ks = k;
    // cout<< " m:" << m << " n:" << n  <<endl;
  
    /* step 1: create cusolver handle, bind a stream */
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
        
    cublas_status = cublasSetMathMode(cublasH, mathMode);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    int truncat_number = h_r_array[0] * h_r_array[1];
    dt *d_A2 = NULL, *d_A2Mid=NULL;
    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(float) * m * n);
    cudaStat2 = cudaMalloc ((void**)&d_U  , sizeof(float) * m * (ks));
    cudaStat3 = cudaMalloc((void**)&d_A2Mid, sizeof(dt)*truncat_number*n);
    cudaStat4 = cudaMalloc ((void**)&d_G1  , sizeof(float) * truncat_number * m);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);

    cudaStat1 = cudaMemcpy(d_A, h_tensor, sizeof(dt)*m*n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    // printMatrix_Device(lad,n,d_A,lad,"A");
    basicRandSVD_LU(cusolverH, cublasH, d_A, m, n, k, s, p, algo, d_U, d_A2Mid);
///计算rsvd获得矩阵 [U S V]
    // basicRandSVD(cusolverH, cublasH, d_A, m, n, k, s, p, algo, d_U, d_S, d_V);

    // h_r_array[1] = k;
    
    // cout << "truncat_number:" << truncat_number << endl;
    dim3 threads(1024,1,1);
    dim3 blocksldur1r2((truncat_number*m+1024-1)/1024,1,1);
    tensorToMode2<<<blocksldur1r2,threads>>>(d_U, d_G1, m, h_r_array[0], h_r_array[1]);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
///根绝截断值将核G1复制到内存
    cudaStat1 = cudaHostAlloc((void**)&h_G1, sizeof(dt)*m*truncat_number, 0);
    cudaStat2 = cudaMemcpyAsync(h_G1, d_G1, sizeof(dt)*m*truncat_number, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    if (d_G1) cudaFree(d_G1); d_G1 = NULL; 
    // if(d_U) cudaFree(d_U); d_U = NULL;
    
///TODO 计算中间变量
    //TODO将s向量变成对角矩阵
    //TODO计算对角矩阵乘V转置矩阵
    
    
    // printMatrix_Device(truncat_number, n, d_VT, truncat_number, "VT2");
    // printMatrix_Device(truncat_number, 1, d_S, truncat_number, "S2");
    // printMatrix_Device(truncat_number, n, d_A2Mid, truncat_number, "A2Mid");

    cudaStat1 =cudaMalloc((void**)&d_A2, sizeof(dt)*truncat_number*n); //56*1000*1000
    assert(cudaStat1 == cudaSuccess);

    dim3 blocksr1r2n((h_r_array[0]*h_r_array[1]*n+1024-1)/1024,1,1);
    tensorToMode231<<<blocksr1r2n,threads>>>(d_A2Mid, d_A2, h_r_array[0], h_r_array[1], n);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);


    if (d_A2Mid) cudaFree(d_A2Mid); d_A2Mid = NULL;
    // if (d_Struncat)  cudaFree(d_Struncat);
   
    // printMatrix_Device(h_r_array[1], n * h_r_array[0], d_A2, h_r_array[1], "A2");
    // printf("step2 - calculate mid success--------2*4--------- \n");

///TODO第二部分计算
    // cusolverH = NULL;
    // cusolver_status = cusolverDnCreate(&cusolverH);
    // assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    dt *d_U2 = NULL;  /* left singular vectors */

    dt *d_G3 = NULL;
    dt *h_G2 = NULL;
    dt *h_G3 = NULL;

    const int m2 = h_n_array[1] * h_r_array[1];  // 56*1000
    const int n2 = h_n_array[2] * h_r_array[0];  // 1000

    k = h_r_array[2];
    truncat_number = h_r_array[2];
    // cout << "truncat_number:" << truncat_number << endl;
    ks = k + s;

    // cout << " m2:" << m2 << "  n2:" << n2 << endl;
 /* step 1: create cusolver handle, bind a stream */
    cudaStat1 = cudaMalloc ((void**)&d_U2   , sizeof(dt)*m2*ks);  // 56*1000*64
    cudaStat2 = cudaMalloc ((void**)&d_G3   , sizeof(dt)*n2*truncat_number);  //1000*56
    cudaStat3 = cudaHostAlloc((void**)&h_G2 , sizeof(dt)*m2*truncat_number, 0); //56*1000*56
    cudaStat4 = cudaHostAlloc((void**)&h_G3 , sizeof(dt)*n2*truncat_number, 0); //1000*56
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);


    // printMatrix_Device(10, 10, d_A2, 10, "A2");
    // cout <<"m2"<< m2 <<"n2"<< n2 <<endl;
    basicRandSVD_LU(cusolverH, cublasH, d_A2, m2, n2, k, s, p, algo, d_U2, d_G3);

    // printMatrix_Device(10, 10, d_U2, 10, "U2");
    // printMatrix_Device(10, 10, d_G3, 10, "G3");
    // basicRandSVD(cusolverH, cublasH, d_A2, m2, n2, k, s, p, algo, d_U2, d_S2, d_V2);
    if(d_A2) cudaFree(d_A2); d_A2 = NULL;


    cudaStat1 = cudaMemcpyAsync(h_G2, d_U2, sizeof(dt)*m2*truncat_number, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpyAsync(h_G3, d_G3, sizeof(dt)*n2*truncat_number, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    // float time2 = timer1.seconds();
    // printf("\n***********************\n time1: %f \n*********************\n", time2);

    if(CalculateError){
        long long calAllNumber = (long long) h_n_array[0]*(long long)h_n_array[1]*(long long)h_n_array[2];
        printf("calAllNumber : %lld\n", calAllNumber);
        double re = 0.0, before = 0.0;
        const float h_one = 1;
        const float h_minus_one = -1;
        before = norm2HH(h_tensor, calAllNumber);

        // dim3 threads(1024,1,1);
        // dim3 block3n((calAllNumber+1024-1)/1024,1,1); 
        float *d_coreG2G3 = NULL, *h_A = NULL, *d_coreG2G3231 = NULL;//, *d_coreG1G2G3 = NULL;
        cudaStat1 = cudaMalloc((void**)&d_coreG2G3, sizeof(float)*m2*n2); // 56*1000*1000
        cudaStat4 = cudaMalloc((void**)&d_coreG2G3231, sizeof(float)*m2*n2); // 56*1000*1000
        cudaStat2 = cudaHostAlloc((void**)&h_A,sizeof(dt)*calAllNumber,0);
        // cudaMalloc((void**)&d_coreG1G2G3, sizeof(float)*calAllNumber); //1000*1000*1000
        cudaStat3 = cudaMemcpy(d_A, h_tensor, sizeof(dt)*calAllNumber, cudaMemcpyHostToDevice);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        assert(cudaStat3 == cudaSuccess);
        assert(cudaStat4 == cudaSuccess);

        cublas_status = cublasGemmEx(cublasH,
                               CUBLAS_OP_N, 
                               CUBLAS_OP_N,
                               m2, n2, truncat_number,
                               &alpha,
                               d_U2, 
                               CUDA_R_32F, 
                               m2,
                               d_G3, 
                               CUDA_R_32F, 
                               truncat_number,
                               &beta,
                               d_coreG2G3231, 
                               CUDA_R_32F, 
                               m2,
                               CUDA_R_32F, 
                               algo);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);
        
        // printMatrix_Device(h_r_array[1], n * h_r_array[0], d_coreG2G3231, h_r_array[1], "CoreG2G3231");
        // change 231 to 123 use mode3
        // dim3 blocksr1r2n((h_r_array[0]*h_r_array[1]*n+1024-1)/1024,1,1);
        tensorToMode3<<<blocksr1r2n,threads>>>(d_coreG2G3231, d_coreG2G3, h_r_array[1], n, h_r_array[0]);
        cudaStat1 = cudaDeviceSynchronize();
        assert(cudaStat1 == cudaSuccess);

        // printMatrix_Device(h_r_array[1]*h_r_array[0], n, d_coreG2G3, h_r_array[1]*h_r_array[0], "CoreG2G3");

        cublas_status = cublasGemmEx(cublasH,
                               CUBLAS_OP_N, 
                               CUBLAS_OP_N,
                               h_n_array[0],  h_n_array[1]* h_n_array[2], h_r_array[0]*h_r_array[1],
                               &h_minus_one,
                               d_U, 
                               CUDA_R_32F, 
                               h_n_array[0],
                               d_coreG2G3, 
                               CUDA_R_32F, 
                               h_r_array[0]*h_r_array[1],
                               &h_one,
                               d_A, 
                               CUDA_R_32F, 
                               h_n_array[0],
                               CUDA_R_32F, 
                               algo);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);
        cudaDeviceSynchronize();
        
        cudaStat2 = cudaMemcpy(h_A, d_A, sizeof(float) * calAllNumber, cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat2);

        // cublasSnrm2(cublasH,calAllNumber,d_coreG1G2G3,1,&re);
        // cublasSnrm2(cublasH,calAllNumber,d_A,1,&  before);
        // cudaDeviceSynchronize();
        re = norm2HH(h_A, calAllNumber);
        cout << "re: "<< re << " before: " << before <<endl;
        printf("\n******************************\n error rate: %E \n*****************************\n", re/before);

        if(d_coreG2G3) cudaFree(d_coreG2G3); d_coreG2G3=NULL;
        if(d_coreG2G3231) cudaFree(d_coreG2G3231); d_coreG2G3231=NULL;
        if (h_A) cudaFreeHost(h_A); h_A = NULL;
    }


    if(d_A) cudaFree(d_A); d_A = NULL;
    if(d_G3) cudaFree(d_G3); d_G3 = NULL;
    if(d_U) cudaFree(d_U); d_U = NULL;
    if (d_U2) cudaFree(d_U2); d_U2 = NULL;

    if (h_G1) cudaFreeHost(h_G1); h_G1 = NULL;
    if (h_G2) cudaFreeHost(h_G2); h_G2 = NULL;
    if (h_G3) cudaFreeHost(h_G3); h_G3 = NULL;

    if (cublasH ) cublasDestroy(cublasH); cublasH = NULL;
    if (cusolverH) cusolverDnDestroy(cusolverH); cusolverH = NULL;
    
    float time = timer0.seconds();
    // cout << "----------------------------------\nall run time: " << time << endl;
    return time;
}