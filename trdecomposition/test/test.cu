#include <iostream>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <time.h>
#include <fstream>
using namespace std;
typedef int lg;
typedef int dt;


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
        return time ;
    }
    private:
    cudaEvent_t start_, stop_;
};

// nvcc test.cu -o test
void printMatrix(int m, int n, const dt*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
        	// if(row == col){
            	float Areg = A[row + col*lda];
            	printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        	// }
        }
    }
    cout << " ------------------------------------" << endl;
}
// nvcc -lcublas -lcusolver -lcurand -std=c++11 test.cu -o test
void printMatrix_Device(int m, int n, const dt*d_A, int lda, const char* name)
{
    dt *h_A = NULL;
    cudaHostAlloc((void**)&h_A , sizeof(dt)*m*n, 1);
    cudaMemcpy(h_A, d_A, sizeof(dt)*m*n, cudaMemcpyDeviceToHost);
    printMatrix(m, n, h_A, lda, name);
    if(h_A) cudaFreeHost(h_A);
}

void printTensor(dt *d_des,long m,long n,long l){
    dt *des = new dt[m*n*l]();
    cudaMemcpy(des,d_des,sizeof(dt)*m*n*l,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int k = 0;k<l;k++){
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                cout<<des[k*m*n+j*m+i]<<" ";
            }
            cout<<endl;
        }
        cout<<"~~~~~~~~~~~~~~~~"<<endl;
    }
    delete[] des;des=nullptr;
}

__global__ void tensorToMode2(dt *T1,dt *T2,int m,int n,int k){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*n*k){
        int tube = i/(m*n);
        int row = (i-tube*(m*n))%m;
        int col = (i-tube*(m*n))/m;
        T2[tube*m*n+row*n+col] = T1[tube*m*n+col*m+row];
    }
    __syncthreads();
}

__global__ void tensorToMode3(dt *T1,dt *T2,int m,int n,int k){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*n*k){
        int tube = i/(m*n);
        int row = (i-tube*(m*n))%m;
        int col = (i-tube*(m*n))/m;
        T2[k*(col*m+row)+tube] = T1[tube*m*n+col*m+row];
    }
    __syncthreads();
}

__global__ void tensorToMode231(dt *T1,dt *T2,int m,int n,int k){
   int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*n*k){
        int tube = i/(m*n);
        int row = (i-tube*(m*n))%m;
        int col = (i-tube*(m*n))/m;
        T2[row*k*n+tube*n+col] = T1[tube*m*n+col*m+row]; //T1[i]
    }
    __syncthreads();
}

int main(int argc, char const *argv[]) {
    int m = 4;
    int n = 3;
    int k = 2;

    int h_A[m*n*k] = {1,2,3,4,5,6,7,8,9,10,11,12,
                13,14,15,16,17,18,19,20,21,22,23,24}; 
    int *d_A=NULL, *d_B = NULL, *d_C = NULL,*d_D = NULL;
    

    cudaMalloc((void**)&d_A, sizeof(int)*m*n*k);
    cudaMalloc((void**)&d_B, sizeof(int)*m*n*k);
    cudaMalloc((void**)&d_C, sizeof(int)*m*n*k);
    // cudaMalloc((void**)&d_D, sizeof(int)*m*n*k);
    cudaMemcpyAsync(d_A, h_A, sizeof(int)*m*n*k, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    printTensor(d_A,m,n,k);

    dim3 thread(512,1,1);
    dim3 block((m*n*k+1024-1)/1024,1,1);
    tensorToMode231<<<block,thread>>>(d_A,d_B,m,n,k);
    cudaDeviceSynchronize();
    printTensor(d_B,n,k,m);

    tensorToMode3<<<block,thread>>>(d_B,d_C,n,k,m);
    cudaDeviceSynchronize();
    printTensor(d_C,m,n,k);

    cudaFreeHost(h_A);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
	return 0;
}	
