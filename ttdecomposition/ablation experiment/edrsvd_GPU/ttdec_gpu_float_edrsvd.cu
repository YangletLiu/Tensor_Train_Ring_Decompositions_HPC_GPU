#include "head.h"

cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT; 
cublasMath_t mathMode = CUBLAS_DEFAULT_MATH; 
cudaError_t cudaStat1 = cudaSuccess;
cudaError_t cudaStat2 = cudaSuccess;
cudaError_t cudaStat3 = cudaSuccess;
cudaError_t cudaStat4 = cudaSuccess;
cudaError_t cudaStat1tat5 = cudaSuccess;
bool CalculateError = false;
const float alpha = 1.0, beta = 0.0;
const double tol = 1.e-12;

double norm2HH(float *A, long long len){
  // printf("%lld\n", len);
  double norm2Result = 0.0;
  for(long long i = 0; i < len; ++i){
    norm2Result += (A[i] * A[i]);
    // if( i % 1000000000 == 0){
    //   printf("%E\n", norm2Result);
    // }
  }
  norm2Result = sqrtf(norm2Result);
  return norm2Result;
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

void basicSvd(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, float *d_B, const int m, const int n, float *d_UT, float *d_S, float *d_V){
    printf("start svd for m: %d, n:%d\n", m,n);
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
    cout << "svd sapce: " << sizeof(float)*lwork<<endl;
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

    basicSvd(cusolverH, cublasH, d_B, ks, n, d_UT, d_S, d_V);

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

    time = timer1.seconds();
    if(d_B) cudaFree(d_B); d_B = NULL;
    if(d_C) cudaFree(d_C); d_C = NULL;
    if(d_UT) cudaFree(d_UT); d_UT = NULL;
}


float ttdec_gpu_float(float* h_tensor, int* h_n_array, int* h_r_array,double *eps, float* singleError, float* singleComRatio ){
    // cout << "algo: " << algo <<endl;
    GPUTimer timer0;
    timer0.start();

    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;

    dt *d_A = NULL;  /* device copy of A */
    dt *d_S = NULL;  /* singular values */
    dt *d_U = NULL;  /* left singular vectors */
    dt *d_V = NULL;  /* right singular vectors */

    dt *h_G1 = NULL;

    const int m = h_n_array[0];  //1500
    const int n = h_n_array[1] * h_n_array[2];  //1500 * 1500 = 2250000
    int k = h_r_array[1];
    const int s = 8;
    const int p = 1;
    int ks = k+s;

    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
        
    cublas_status = cublasSetMathMode(cublasH, mathMode);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    
    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(float) * m * n);
    cudaStat2 = cudaMalloc ((void**)&d_U  , sizeof(float) * m * (ks));
    cudaStat3 = cudaMalloc ((void**)&d_S  , sizeof(float) * (ks));
    cudaStat4 = cudaMalloc ((void**)&d_V  , sizeof(float) * n * (ks));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    // cudaStat1 = cudaMemcpy(d_A, h_tensor, sizeof(dt)*m*n, cudaMemcpyHostToDevice);
    // assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMemcpyAsync(d_A, h_tensor, sizeof(dt)*m*n, cudaMemcpyHostToDevice);
    cudaStat2 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    // printMatrix_Device(lad,n,d_A,lad,"A");
    
///计算rsvd获得矩阵 [U S V]
    basicRandSVD(cusolverH, cublasH, d_A, m, n, k, s, p, algo, d_U, d_S, d_V);

    // h_r_array[1] = k;
    int truncat_number = h_r_array[1];
    // cout << "truncat_number:" << truncat_number << endl;

///根绝截断值将核G1复制到内存
    cudaStat1 = cudaHostAlloc((void**)&h_G1, sizeof(dt)*m*truncat_number, 0);
    cudaStat2 = cudaMemcpyAsync(h_G1, d_U, sizeof(dt)*m*truncat_number, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    // if (d_U) cudaFree(d_U);
    
///TODO 计算中间变量
    //TODO将s向量变成对角矩阵
    //TODO计算对角矩阵乘V转置矩阵
    dt *d_A2 = NULL, *d_VT = NULL;
    cudaStat1 =cudaMalloc((void**)&d_A2, sizeof(dt)*truncat_number*n); //56*1000*1000
    cudaStat2 =cudaMalloc((void**)&d_VT, sizeof(dt)*ks*n);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    cudaDeviceSynchronize();

    cublas_status = cublasSgeam(cublasH,
                                CUBLAS_OP_T, CUBLAS_OP_N, 
                                ks, n,
                                &alpha,
                                d_V, n,
                                &beta,
                                d_V, n,
                                d_VT, ks);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaDeviceSynchronize();
    // printMatrix_Device(ks,n,d_VT,ks,"VT");
    if (d_V) cudaFree(d_V); d_V=NULL;
//再计算矩阵  列向量乘法  k*k  k*n
    cublas_status = cublasSdgmm(cublasH,
                                CUBLAS_SIDE_LEFT,
                                truncat_number, n,
                                d_VT, ks,
                                d_S, 1,
                                d_A2, truncat_number);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaDeviceSynchronize();

    if (d_VT) cudaFree(d_VT);
    if (d_S) cudaFree(d_S);
    // if (d_Struncat)  cudaFree(d_Struncat);
    d_S = NULL;
    d_VT = NULL;

    dt *d_S2 = NULL;  /* singular values */
    dt *d_U2 = NULL;  /* left singular vectors */
    dt *d_V2 = NULL;  /* right singular vectors */
    dt *d_VT2 = NULL;

    dt *d_G3 = NULL;
    dt *h_G2 = NULL;
    dt *h_G3 = NULL;

    const int m2 = h_n_array[1] * h_r_array[1];  // 56*1000
    const int n2 = h_n_array[2];  // 1000

    k = h_r_array[2];
    truncat_number = h_r_array[2];
    ks = k + s;

 /* step 1: create cusolver handle, bind a stream */
    cudaStat1 = cudaMalloc ((void**)&d_U2   , sizeof(dt)*m2*ks);  // 56*1000*64
    cudaStat2 = cudaMalloc ((void**)&d_S2   , sizeof(dt)*ks);  // 64 
    cudaStat3 = cudaMalloc ((void**)&d_V2   , sizeof(dt)*n2*ks);  // 1000*64
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    basicRandSVD(cusolverH, cublasH, d_A2, m2, n2, k, s, p, algo, d_U2, d_S2, d_V2);
    if(d_A2) cudaFree(d_A2); d_A2 = NULL;

    // h_r_array[2] = k;
    cudaStat1 = cudaMalloc ((void**)&d_VT2  , sizeof(dt)*ks*n2);  //64*1000
    cudaStat2 = cudaMalloc ((void**)&d_G3   , sizeof(dt)*n2*truncat_number);  //1000*56
    cudaStat3 = cudaHostAlloc((void**)&h_G2 , sizeof(dt)*m2*truncat_number, 0); //56*1000*56
    cudaStat4 = cudaHostAlloc((void**)&h_G3 , sizeof(dt)*n2*truncat_number, 0); //1000*56
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);

    //对V2进行转置
    cublas_status = cublasSgeam(cublasH,
                                CUBLAS_OP_T, CUBLAS_OP_N, 
                                ks, n2,
                                &alpha,
                                d_V2, n2,
                                &beta,
                                d_V2, n2,
                                d_VT2, ks);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);

    cublas_status = cublasSdgmm(cublasH,
                                CUBLAS_SIDE_LEFT,
                                truncat_number, n2,
                                d_VT2, ks,
                                d_S2, 1,
                                d_G3, truncat_number);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpyAsync(h_G2, d_U2, sizeof(dt)*m2*truncat_number, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpyAsync(h_G3, d_G3, sizeof(dt)*n2*truncat_number, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

if(CalculateError){
    long long calAllNumber = (long long) h_n_array[0]*(long long)h_n_array[1]*(long long)h_n_array[2];
    printf("calAllNumber : %lld\n", calAllNumber);
    double re = 0.0, before = 0.0;
    const float h_one = 1;
    const float h_minus_one = -1;
    before = norm2HH(h_tensor, calAllNumber);

    dim3 threads(1024,1,1);
    dim3 block3n((calAllNumber+1024-1)/1024,1,1); 
    float *d_coreG2G3 = NULL, *h_A = NULL;//, *d_coreG1G2G3 = NULL;
    cudaStat1 = cudaMalloc((void**)&d_coreG2G3, sizeof(float)*m2*n2); // 56*1000*1000
    cudaStat2 = cudaHostAlloc((void**)&h_A,sizeof(dt)*calAllNumber,0);
    cudaStat3 = cudaMemcpy(d_A, h_tensor, sizeof(dt)*calAllNumber, cudaMemcpyHostToDevice);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);

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
                           d_coreG2G3, 
                           CUDA_R_32F, 
                           m2,
                           CUDA_R_32F, 
                           algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    cublas_status = cublasGemmEx(cublasH,
                           CUBLAS_OP_N, 
                           CUBLAS_OP_N,
                           h_n_array[0],  h_n_array[1]* h_n_array[2], h_r_array[1],
                           &h_minus_one,
                           d_U, 
                           CUDA_R_32F, 
                           h_n_array[0],
                           d_coreG2G3, 
                           CUDA_R_32F, 
                           h_r_array[1],
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

    re = norm2HH(h_A, calAllNumber);
    cout << "re: "<< re << " before: " << before <<endl;
    printf("\n******************************\n error rate: %E \n*****************************\n", re/before);

    if(d_coreG2G3) cudaFree(d_coreG2G3); d_coreG2G3=NULL;
    // if(d_coreG1G2G3) cudaFree(d_coreG1G2G3); d_coreG1G2G3=NULL;
    if (h_A) cudaFreeHost(h_A); h_A = NULL;
}


    if(d_A) cudaFree(d_A); d_A = NULL;
    if(d_G3) cudaFree(d_G3); d_G3 = NULL;
    if(d_U) cudaFree(d_U); d_U = NULL;

    if (d_S2) cudaFree(d_S2); d_S2 = NULL;
    if (d_U2) cudaFree(d_U2); d_U2 = NULL;
    if (d_V2) cudaFree(d_V2); d_V2 = NULL;
    if (d_VT2) cudaFree(d_VT2); d_VT2 = NULL;

    if (h_G1) cudaFreeHost(h_G1); h_G1 = NULL;
    if (h_G2) cudaFreeHost(h_G2); h_G2 = NULL;
    if (h_G3) cudaFreeHost(h_G3); h_G3 = NULL;

    if (cublasH ) cublasDestroy(cublasH); cublasH = NULL;
    if (cusolverH) cusolverDnDestroy(cusolverH); cusolverH = NULL;
   
    float time = timer0.seconds();
    // cout << "----------------------------------\nall run time: " << time << endl;
    return time;
}