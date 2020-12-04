/*************************************************************************
	> File Name: func.h
	> Author: honghao
	> Mail: honghao@shu.edu.cn 
	> Created Time: 2020年04月2日 星期四 14时25分17秒
 ************************************************************************/
#include "head.h"


//进行3维的 稠密  张量分解
int main(){
    const int k = 48;
    int calculateTimes = 10;
    if(calError)
        calculateTimes = 1;
    int mid_rank = 4;
    if(k > 24){
        mid_rank = 8;
    }
    int ttRanks[7] = {1,mid_rank,mid_rank,mid_rank,mid_rank,mid_rank,1};
    int ttDemns[6] = {k, k, k, k, k, k};

    const int p = 8*8;
    int m = k;
    long long n = pow(k,5);
    printf("%lld , mid_rank = %d, k = %d \n", sizeof(dt)*m*n, mid_rank,k);
    srand(1134);
    dt *h_A = NULL;
    cudaStat1 = cudaHostAlloc((void**)&h_A,sizeof(dt)*m*n,0);
    assert(cudaStat1 == cudaSuccess);
    for(long long i = 0; i < n*m; i++){
        h_A[i] = (dt) ((rand()*1.0) / (RAND_MAX*1.0))*2.0 - 1.0;
    }

    warmupcu();
    // ttdec_half_lu_host_6(h_A, ttRanks, ttDemns, 1, p);

    float time = ttdec_half_lu_host_6(h_A, ttRanks, ttDemns, calculateTimes, p);

    printf("*************time****************\n %f \n*******************************\n", time);
 
    if(h_A      ) cudaFreeHost(h_A); h_A = NULL;  

    return 0;
}