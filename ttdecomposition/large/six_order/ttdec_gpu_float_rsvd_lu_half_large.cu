#include "head.h"

bool reduceDim =true;
bool calError = true;
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
cudaError_t cudaStat7 = cudaSuccess;
const float alpha = 1.0, beta0 = 0.0, beta_1 = -1.0, beta1 = 1.0;


//在TT分解中返回的是U 然后利用
void basicEig(cusolverDnHandle_t cusolverH, dt *d_A, const int &m, dt *d_W){  
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

    // printMatrix_Device(10, 10, d_A, 10, "A");
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

    // printMatrix_Device(10, 10, d_A, 10, "A2");
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("after syevd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    // if (d_W    ) cudaFree(d_W);
}

void largeMatrixSelfMulti_once_device(cublasHandle_t cublasH, dt *d_A,const int &m,const long long &n, dt *d_AAT){
    half *d_Ah = NULL;
    cudaStat1 = cudaMalloc((void**)&d_Ah, sizeof(half)*m*n);
    assert(cudaStat1 == cudaSuccess);
    f2h(d_A, d_Ah, m*n);
    cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                           m, m, n,
                           &alpha, d_Ah, CUDA_R_16F, m,
                           d_Ah, CUDA_R_16F, m,
                           &beta1, d_AAT, CUDA_R_32F, m,
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    if(d_Ah     ) cudaFree(d_Ah); d_Ah=NULL;
}

void smallargeMatrixMulti_once_device(cublasHandle_t cublasH, dt *d_A, dt *d_B, const int &ttrank, const int &m, const long long &n, dt *d_ATB){
    half *d_Ah = NULL, *d_Bh = NULL;
    cudaStat1 = cudaMalloc((void**)&d_Ah, sizeof(half)*m*ttrank);
    cudaStat2 = cudaMalloc((void**)&d_Bh, sizeof(half)*m*n);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    f2h(d_A, d_Ah, m*ttrank);
    f2h(d_B, d_Bh, m*n);

    cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                           ttrank, n, m,
                           &alpha, d_Ah, CUDA_R_16F, m,
                           d_Bh, CUDA_R_16F, m,
                           &beta0, d_ATB, CUDA_R_32F, ttrank,
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    if(d_Ah     ) cudaFree(d_Ah); d_Ah=NULL;
    if(d_Bh     ) cudaFree(d_Bh); d_Bh=NULL;
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

void longMatrixSVD_Eign_once_device(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, dt *d_A, const int &m, const long long &n, const int &ttRank, dt *d_G, dt *d_A2, const int &p){
    dt *d_W=NULL, *d_AAT=NULL;
    cudaStat1 = cudaMalloc((void**)&d_AAT, sizeof(dt)*m*m);
    cudaStat2 = cudaMalloc((void**)&d_W, sizeof(dt)*m);

     // printMatrix_Device( m, n, d_A, m, "A");
    largeMatrixSelfMulti_once_device(cublasH, d_A, m, n, d_AAT);
    // printMatrix_Device( m, m, d_AAT, m, "AAT");
    basicEig(cusolverH, d_AAT, m, d_W);
     // printMatrix_Device( m, m, d_AAT, m, "AAT");
    if(d_W     ) cudaFree(d_W); d_W = NULL;
    matrixInvertColumn(d_AAT, d_G, m, ttRank);
    if(d_AAT    ) cudaFree(d_AAT);d_AAT=NULL;
    // printMatrix_Device( 10,10, d_G1, 10, "G1");
    // 大于24需要分片
    if(m>24 && n>pow(m,4)){
        smallargeMatrixMulti_slice_device(cublasH, d_G, d_A, ttRank, m, n, d_A2, p);
        // printMatrix_Device(10, 10, d_A2, 10, "A2");
    }
    else
        smallargeMatrixMulti_once_device(cublasH, d_G, d_A, ttRank, m, n, d_A2);
    // printMatrix_Device(10, 10, d_A2, 10, "A2");
}

//TODO 实现一个超大规模矩阵A的 A*AT，存储到GPU上
void largeMatrixSelfMulti_slice_host(cublasHandle_t cublasH, dt *h_A,const int &m,const long long &n, dt *d_AAT,const int &p){
    long long slice = 0, le = 0;
    if(n%p==0){
        slice = n/p;
    }else{
        slice = n/p;
        le = n%p;
    }
    float *d_tempA =NULL; 
    half *d_Ah = NULL;
    cudaStat1 = cudaMalloc((void**)&d_tempA, sizeof(dt)*m*slice);
    cudaStat2 = cudaMalloc((void**)&d_Ah, sizeof(half)*m*slice);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    // printf("%d\n", slice);
    for(int i=0;i<p;i++){
        cudaStat3 = cudaMemcpyAsync(d_tempA, h_A+m*slice*i, sizeof(dt)*m*slice, cudaMemcpyHostToDevice,0);
        cudaStat4 = cudaDeviceSynchronize();
        assert(cudaStat3 == cudaSuccess);
        assert(cudaStat4 == cudaSuccess);
        //判断是否降维
        // printMatrix_Device(m, slice, d_tempA, m, "d_tempA");
        f2h(d_tempA, d_Ah, m*slice);
        cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                           m, m, slice,
                           &alpha, d_Ah, CUDA_R_16F, m,
                           d_Ah, CUDA_R_16F, m,
                           &beta1, d_AAT, CUDA_R_32F, m,
                           CUDA_R_32F, algo);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    }
    if(d_Ah     ) cudaFree(d_Ah); d_Ah=NULL;
    if(d_tempA  ) cudaFree(d_tempA); d_tempA=NULL;
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

void longMatrixSVD_Eign_once_host(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, dt *h_A, const int &m, const long long &n, const int &ttRank, dt *h_G, dt *h_A2, const int &p){
    dt *d_W=NULL, *d_AAT=NULL,*d_G=NULL;
    cudaStat1 = cudaMalloc((void**)&d_AAT, sizeof(dt)*m*m);
    cudaStat2 = cudaMalloc((void**)&d_W, sizeof(dt)*m);
    cudaStat3 = cudaMalloc((void**)&d_G, sizeof(dt)*m*ttRank);

    // printf("selfMtrixMul\n");
    // printMatrix_Device( m, n, d_A, m, "A");
    largeMatrixSelfMulti_slice_host(cublasH, h_A, m, n, d_AAT, p);
    // printMatrix_Device( m, m, d_AAT, m, "AAT");
    // printf("basicEig\n");
    basicEig(cusolverH, d_AAT, m, d_W);
     // printMatrix_Device( m, m, d_AAT, m, "AAT");
    if(d_W     ) cudaFree(d_W); d_W = NULL;
    // printf("matrixInvertColumn\n");
    matrixInvertColumn(d_AAT, d_G, m, ttRank);
    if(d_AAT    ) cudaFree(d_AAT);d_AAT=NULL;
    // printMatrix_Device( 10,10, d_G1, 10, "G1");
    // 大于24需要分片

    // printf("smallargeMatrixMulti_slice_host\n");
    smallargeMatrixMulti_slice_host(cublasH, d_G, h_A, ttRank, m, n, h_A2, p);
    // printMatrix_Device(10, 10, d_A2, 10, "A2");

    cudaStat1 = cudaMemcpyAsync(h_G, d_G, sizeof(dt)*m*ttRank, cudaMemcpyDeviceToHost,0);
    assert(cudaStat1 == cudaSuccess);
    if(d_G) cudaFree(d_G);d_G=NULL;
    // printMatrix_Device(10, 10, d_A2, 10, "A2");

}


float ttdec_half_lu_host_6(dt* h_A, int *ttRanks, const int *ttDemns, const int &calculateTimes, const int &p){
    printf("Start mul \n");
    GPUTimer timer;
    timer.start();

    for(int i=0;i<calculateTimes;i++){
        printf("***************************** %d ********************************\n", i);
        cublasHandle_t cublasH = NULL;
        cusolverDnHandle_t cusolverH = NULL;

        cusolver_status = cusolverDnCreate(&cusolverH);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

        cublas_status = cublasCreate(&cublasH);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        cublas_status = cublasSetMathMode(cublasH, mathMode);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
        
    //1
        int m = ttDemns[0];
        int n = ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5];

        dt *h_A1=NULL, *h_G1=NULL;
        cudaStat1 = cudaHostAlloc((void**)&h_G1,sizeof(dt)*m*ttRanks[1],0);
        cudaStat1 = cudaHostAlloc((void**)&h_A1,sizeof(dt)*n*ttRanks[1],0);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(m, n, d_A, m, "A");
        longMatrixSVD_Eign_once_host(cublasH, cusolverH, h_A, m, n, ttRanks[1] , h_G1, h_A1, p);
        // printMatrix(10, 10, h_G1, 10, "G1");
        // printMatrix(10, 10, h_A2, 10, "A2");

        dt *d_A2=NULL;
        cudaStat1 = cudaMalloc((void**)&d_A2, sizeof(dt)*n*ttRanks[1]);
        cudaStat2 = cudaMemcpyAsync(d_A2, h_A1, sizeof(dt)*n*ttRanks[1], cudaMemcpyHostToDevice,0);
        // cudaStat3 = cudaDeviceSynchronize();
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        if(h_A1     ) cudaFreeHost(h_A1);h_A1 = NULL;

    //1->2
        dt *d_G2=NULL, *d_A3=NULL, *h_G2=NULL;
        m = ttRanks[1] * ttDemns[1];
        n = n/ttDemns[1];
        cudaStat1 = cudaMalloc((void**)&d_G2, sizeof(dt)*m*ttRanks[2]);
        cudaStat2 = cudaHostAlloc((void**)&h_G2,sizeof(dt)*m*ttRanks[2],0);
        cudaStat2 = cudaMalloc((void**)&d_A3, sizeof(dt)*n*ttRanks[2]);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(ttRanks[1]*m, n, d_A2, ttRanks[1]*m, "A22");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A2, m, n, ttRanks[2] , d_G2, d_A3, p);
        // printMatrix_Device(ttRanks[1]*m, ttRanks[2], d_G2, ttRanks[1]*m, "G22");
        // printMatrix_Device(ttRanks[2], n, d_A3, ttRanks[2], "A32");
        if(d_A2     ) cudaFree(d_A2); d_A2 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G2, d_G2, sizeof(dt)*m*ttRanks[2], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G2     ) cudaFree(d_G2); d_G2 = NULL;

    //1->2->3
        dt *d_G3=NULL, *d_A4=NULL, *h_G3=NULL;
        m = ttRanks[2] * ttDemns[2];
        n = n/ttDemns[2];
        cudaStat1 = cudaMalloc((void**)&d_G3, sizeof(dt)*m*ttRanks[3]);
        cudaStat2 = cudaHostAlloc((void**)&h_G3,sizeof(dt)*m*ttRanks[3],0);
        cudaStat2 = cudaMalloc((void**)&d_A4, sizeof(dt)*ttRanks[3]*n);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(ttRanks[1]*m, n, d_A2, ttRanks[1]*m, "A22");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A3, m, n, ttRanks[3] , d_G3, d_A4, p);
        // printMatrix_Device(ttRanks[1]*m, ttRanks[2], d_G2, ttRanks[1]*m, "G22");
        // printMatrix_Device(ttRanks[2], n, d_A3, ttRanks[2], "A32");
        if(d_A3     ) cudaFree(d_A3); d_A3 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G3, d_G3, sizeof(dt)*m*ttRanks[3], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G3     ) cudaFree(d_G3); d_G3 = NULL;

    //1->2->3->4
        dt *d_G4=NULL, *d_A5=NULL, *h_G4=NULL;
        m = ttRanks[3] * ttDemns[3];
        n = n/ttDemns[3];
        cudaStat1 = cudaMalloc((void**)&d_G4, sizeof(dt)*m*ttRanks[4]);
        cudaStat2 = cudaHostAlloc((void**)&h_G4,sizeof(dt)*m*ttRanks[4],0);
        cudaStat2 = cudaMalloc((void**)&d_A5, sizeof(dt)*ttRanks[4]*n);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(ttRanks[1]*m, n, d_A2, ttRanks[1]*m, "A22");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A4, m, n, ttRanks[4] , d_G4, d_A5, p);
        // printMatrix_Device(ttRanks[1]*m, ttRanks[2], d_G2, ttRanks[1]*m, "G22");
        // printMatrix_Device(ttRanks[2], n, d_A3, ttRanks[2], "A32");
        if(d_A4     ) cudaFree(d_A4); d_A4 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G4, d_G4, sizeof(dt)*m*ttRanks[4], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G4     ) cudaFree(d_G4); d_G4 = NULL;

    //1->2->3->4->5
        dt *d_G5=NULL, *d_G6=NULL, *h_G5=NULL, *h_G6=NULL;
        m = ttRanks[4] * ttDemns[4];
        n = n/ttDemns[4];
        cudaStat1 = cudaMalloc((void**)&d_G5, sizeof(dt)*m*ttRanks[5]);
        cudaStat2 = cudaHostAlloc((void**)&h_G5,sizeof(dt)*m*ttRanks[5],0);
        cudaStat2 = cudaMalloc((void**)&d_G6, sizeof(dt)*ttRanks[5]*n);
        cudaStat2 = cudaHostAlloc((void**)&h_G6,sizeof(dt)*n*ttRanks[5],0);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(ttRanks[1]*m, n, d_A2, ttRanks[1]*m, "A22");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A5, m, n, ttRanks[5] , d_G5, d_G6, p);
        // printMatrix_Device(ttRanks[1]*m, ttRanks[2], d_G2, ttRanks[1]*m, "G22");
        // printMatrix_Device(ttRanks[5], n, d_G6, ttRanks[5], "G6");
        if(d_A5     ) cudaFree(d_A5); d_A5 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G5, d_G5, sizeof(dt)*m*ttRanks[5], cudaMemcpyDeviceToHost,0);
        cudaStat6 = cudaMemcpyAsync(h_G6, d_G6, sizeof(dt)*n*ttRanks[5], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        assert(cudaStat6 == cudaSuccess);
        if(d_G5     ) cudaFree(d_G5); d_G5 = NULL;
        if(d_G6     ) cudaFree(d_G6); d_G6 = NULL;


        if(h_G1     ) cudaFreeHost(h_G1);h_G1 = NULL;
        if(h_G2     ) cudaFreeHost(h_G2);h_G2 = NULL;
        if(h_G3     ) cudaFreeHost(h_G3);h_G3 = NULL;
        if(h_G4     ) cudaFreeHost(h_G4);h_G4 = NULL;
        if(h_G5     ) cudaFreeHost(h_G5);h_G5 = NULL;
        if(h_G6     ) cudaFreeHost(h_G6);h_G6 = NULL;
        if(cublasH  ) cublasDestroy(cublasH);
        if(cusolverH) cusolverDnDestroy(cusolverH); 
    }

    float time = timer.seconds()/calculateTimes;
    return time;
}

double calMSE(cublasHandle_t cublasH, dt *h_A, dt *h_G1, dt *h_G2, dt *h_G3, dt *h_G4, dt *h_G5, dt *h_G6, const int *ttRanks, const int *ttDemns){
    dt *d_G6=NULL, *d_G5=NULL, *d_G56=NULL;
    cudaStat1 = cudaMalloc((void**)&d_G6, sizeof(dt)*ttRanks[5]*ttDemns[5]*ttRanks[6]);
    cudaStat2 = cudaMalloc((void**)&d_G5, sizeof(dt)*ttRanks[4]*ttDemns[4]*ttRanks[5]);
    cudaStat3 = cudaMalloc((void**)&d_G56, sizeof(dt)*ttRanks[4]*ttDemns[4]*ttDemns[5]*ttRanks[6]);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    cudaStat1 = cudaMemcpy(d_G6, h_G6, sizeof(dt)*ttRanks[5]*ttDemns[5]*ttRanks[6], cudaMemcpyHostToDevice);
    cudaStat1 = cudaMemcpy(d_G5, h_G5, sizeof(dt)*ttRanks[4]*ttDemns[4]*ttRanks[5], cudaMemcpyHostToDevice);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    cublas_status = cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                           ttRanks[4]*ttDemns[4], ttDemns[5]*ttRanks[6], ttRanks[5],
                           &alpha, d_G5, ttRanks[5],
                           d_G6, ttRanks[4]*ttDemns[4],
                           &beta0, d_G56, ttRanks[4]*ttDemns[4]); 

}

float ttdec_half_lu_device_6(dt* h_A, const int *ttRanks, const int *ttDemns, const int &calculateTimes, const int &p){
    printf("Start mul \n");
    GPUTimer timer;
    timer.start();
    for(int i=0;i<calculateTimes;i++){
        printf("***************************** %d ********************************\n", i);
        cublasHandle_t cublasH = NULL;
        cusolverDnHandle_t cusolverH = NULL;

        cusolver_status = cusolverDnCreate(&cusolverH);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

        cublas_status = cublasCreate(&cublasH);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        cublas_status = cublasSetMathMode(cublasH, mathMode);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
        

        int m = ttDemns[0];
        int n = ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5];
        dt *d_A=NULL, *d_G1=NULL, *d_A2=NULL, *h_G1=NULL;
        cudaStat1 = cudaMalloc((void**)&d_A, sizeof(dt)*m*n);
        cudaStat2 = cudaHostAlloc((void**)&h_G1,sizeof(dt)*m*ttRanks[1],0);
        cudaStat3 = cudaMalloc((void**)&d_G1, sizeof(dt)*m*ttRanks[1]);
        cudaStat4 = cudaMalloc((void**)&d_A2, sizeof(dt)*n*ttRanks[1]);
        cudaStat5 = cudaMemcpyAsync(d_A, h_A, sizeof(dt)*m*n, cudaMemcpyHostToDevice,0);
        cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        assert(cudaStat3 == cudaSuccess);
        assert(cudaStat4 == cudaSuccess);
        assert(cudaStat5 == cudaSuccess);
        assert(cudaStat6 == cudaSuccess);
        // printMatrix_Device(m, n, d_A, m, "A");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A, m, n, ttRanks[1] , d_G1, d_A2, p);
        // printMatrix_Device(m, ttRanks[1], d_G1, m, "G1");
        // printMatrix_Device(ttRanks[1], n, d_A2, ttRanks[1], "A2");
        if(d_A      ) cudaFree(d_A); d_A = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G1, d_G1, sizeof(dt)*m*ttRanks[1], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G1     ) cudaFree(d_G1); d_G1 = NULL;

    //1->2
        dt *d_G2=NULL, *d_A3=NULL, *h_G2=NULL;
        m = ttRanks[1] * ttDemns[1];
        n = n/ttDemns[1];
        cudaStat1 = cudaMalloc((void**)&d_G2, sizeof(dt)*m*ttRanks[2]);
        cudaStat2 = cudaHostAlloc((void**)&h_G2,sizeof(dt)*m*ttRanks[2],0);
        cudaStat2 = cudaMalloc((void**)&d_A3, sizeof(dt)*n*ttRanks[2]);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(ttRanks[1]*m, n, d_A2, ttRanks[1]*m, "A22");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A2, m, n, ttRanks[2] , d_G2, d_A3, p);
        // printMatrix_Device(ttRanks[1]*m, ttRanks[2], d_G2, ttRanks[1]*m, "G22");
        // printMatrix_Device(ttRanks[2], n, d_A3, ttRanks[2], "A32");
        if(d_A2     ) cudaFree(d_A2); d_A2 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G2, d_G2, sizeof(dt)*m*ttRanks[2], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G2     ) cudaFree(d_G2); d_G2 = NULL;

    //1->2->3
        dt *d_G3=NULL, *d_A4=NULL, *h_G3=NULL;
        m = ttRanks[2] * ttDemns[2];
        n = n/ttDemns[2];
        cudaStat1 = cudaMalloc((void**)&d_G3, sizeof(dt)*m*ttRanks[3]);
        cudaStat2 = cudaHostAlloc((void**)&h_G3,sizeof(dt)*m*ttRanks[3],0);
        cudaStat2 = cudaMalloc((void**)&d_A4, sizeof(dt)*ttRanks[3]*n);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(ttRanks[1]*m, n, d_A2, ttRanks[1]*m, "A22");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A3, m, n, ttRanks[3] , d_G3, d_A4, p);
        // printMatrix_Device(ttRanks[1]*m, ttRanks[2], d_G2, ttRanks[1]*m, "G22");
        // printMatrix_Device(ttRanks[2], n, d_A3, ttRanks[2], "A32");
        if(d_A3     ) cudaFree(d_A3); d_A3 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G3, d_G3, sizeof(dt)*m*ttRanks[3], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G3     ) cudaFree(d_G3); d_G3 = NULL;

    //1->2->3->4
        dt *d_G4=NULL, *d_A5=NULL, *h_G4=NULL;
        m = ttRanks[3] * ttDemns[3];
        n = n/ttDemns[3];
        cudaStat1 = cudaMalloc((void**)&d_G4, sizeof(dt)*m*ttRanks[4]);
        cudaStat2 = cudaHostAlloc((void**)&h_G4,sizeof(dt)*m*ttRanks[4],0);
        cudaStat2 = cudaMalloc((void**)&d_A5, sizeof(dt)*ttRanks[4]*n);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(ttRanks[1]*m, n, d_A2, ttRanks[1]*m, "A22");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A4, m, n, ttRanks[4] , d_G4, d_A5, p);
        // printMatrix_Device(ttRanks[1]*m, ttRanks[2], d_G2, ttRanks[1]*m, "G22");
        // printMatrix_Device(ttRanks[2], n, d_A3, ttRanks[2], "A32");
        if(d_A4     ) cudaFree(d_A4); d_A4 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G4, d_G4, sizeof(dt)*m*ttRanks[4], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G4     ) cudaFree(d_G4); d_G4 = NULL;

    //1->2->3->4->5
        dt *d_G5=NULL, *d_G6=NULL, *h_G5=NULL, *h_G6=NULL;
        m = ttRanks[4] * ttDemns[4];
        n = n/ttDemns[4];
        cudaStat1 = cudaMalloc((void**)&d_G5, sizeof(dt)*m*ttRanks[5]);
        cudaStat2 = cudaHostAlloc((void**)&h_G5,sizeof(dt)*m*ttRanks[5],0);
        cudaStat2 = cudaMalloc((void**)&d_G6, sizeof(dt)*ttRanks[5]*n);
        cudaStat2 = cudaHostAlloc((void**)&h_G6,sizeof(dt)*n*ttRanks[5],0);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(ttRanks[1]*m, n, d_A2, ttRanks[1]*m, "A22");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A5, m, n, ttRanks[5] , d_G5, d_G6, p);
        // printMatrix_Device(ttRanks[1]*m, ttRanks[2], d_G2, ttRanks[1]*m, "G22");
        // printMatrix_Device(ttRanks[5], n, d_G6, ttRanks[5], "G6");
        if(d_A5     ) cudaFree(d_A5); d_A5 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G5, d_G5, sizeof(dt)*m*ttRanks[5], cudaMemcpyDeviceToHost,0);
        cudaStat6 = cudaMemcpyAsync(h_G6, d_G6, sizeof(dt)*n*ttRanks[5], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        assert(cudaStat6 == cudaSuccess);
        if(d_G5     ) cudaFree(d_G5); d_G5 = NULL;
        if(d_G6     ) cudaFree(d_G6); d_G6 = NULL;


        if(h_G1     ) cudaFreeHost(h_G1);h_G1 = NULL;
        if(h_G2     ) cudaFreeHost(h_G2);h_G2 = NULL;
        if(h_G3     ) cudaFreeHost(h_G3);h_G3 = NULL;
        if(h_G4     ) cudaFreeHost(h_G4);h_G4 = NULL;
        if(h_G5     ) cudaFreeHost(h_G5);h_G5 = NULL;
        if(h_G6     ) cudaFreeHost(h_G6);h_G6 = NULL;
        if(cublasH  ) cublasDestroy(cublasH);
        if(cusolverH) cusolverDnDestroy(cusolverH); 
    }

    float time = timer.seconds()/calculateTimes;
    return time;
}