/*************************************************************************
	> File Name: func.h
	> Author: honghao
	> Mail: honghao@shu.edu.cn 
	> Created Time: 2020年04月2日 星期四 14时25分17秒
 ************************************************************************/
#ifndef GUARD_func_h
#define GUARD_func_h

typedef float dt;


void warmupcu();
double norm2HH(float *A, long long len);
void norm2HH_device(float *d_A, long long len, float *norm2);
void matrixInvertColumn(dt *d_A, dt *d_Ainv,const int &m, const int &n);
void printMatrix(int m, int n, const dt*A, int lda, const char* name);
void printMatrix_Device(int m, int n, const dt*d_A, int lda, const char* name);
void f2h(dt *A,half *B,long num);


void basicEig(cusolverDnHandle_t cusolverH, dt *d_A, const int &m, dt *d_W);
void largeMatrixSelfMulti_once_device(cublasHandle_t cublasH, dt *d_A,const int &m,const long long &n, dt *d_AAT);
void smallargeMatrixMulti_once_device(cublasHandle_t cublasH, dt *d_A, dt *d_B, const int &ttrank, const int &m, const long long &n, dt *d_ATB);
void smallargeMatrixMulti_slice_device(cublasHandle_t cublasH, dt *d_A, dt *d_B, const int &ttrank, const int &m, const long long &n, dt *d_ATB, const int &p);
void longMatrixSVD_Eign_once_device(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, dt *d_A, const int &m, const long long &n, const int &ttRank, dt *d_G, dt *d_A2, const int &p);
void largeMatrixSelfMulti_slice_host(cublasHandle_t cublasH, dt *h_A,const int &m,const long long &n, dt *d_AAT,const int &p);
void smallargeMatrixMulti_slice_host(cublasHandle_t cublasH, dt *d_A, dt *h_B, const int &ttrank, const int &m,  const long long &n, dt *h_ATB, const int &p);
void longMatrixSVD_Eign_once_host(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, dt *h_A, const int &m, const long long &n, const int &ttRank, dt *h_G, dt *h_A2, const int &p);
float ttdec_half_lu_host_6(dt* h_A, int *ttRanks, const int *ttDemns, const int &calculateTimes, const int &p);
double calMSE(cublasHandle_t cublasH, dt *h_A, dt *h_G1, dt *h_G2, dt *h_G3, dt *h_G4, dt *h_G5, dt *h_G6, const int *ttRanks, const int *ttDemns);
float ttdec_half_lu_device_6(dt* h_A, const int *ttRanks, const int *ttDemns, const int &calculateTimes, const int &p);


__global__ void warmup();
__global__ void sub(dt *A,dt *B,long a,long b,long c);
__global__  void floattohalf(dt *AA,half *BB,long m);
__global__ void matrixInvertColumn_kernel(dt *d_A, dt *d_Ainv, int m, int n);

#endif
