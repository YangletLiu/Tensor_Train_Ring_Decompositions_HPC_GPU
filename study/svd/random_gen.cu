#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include "curand_kernel.h" 
#include "curand.h"
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <time.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

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


__global__ void kernel_set_random(curandState *curand_states,int width,int height)//,long clock_for_rand)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;

    if(x<0 || x>width)
    {
        return;
    }
    curand_init(1234,x,0,&curand_states[x]);
}

__global__ void kernel_random(float *dev_random_array,int width,int height,curandState *curand_states)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;

    if(x<0 || x>width)
    {
        return;
    }

    for(int y=0;y<height;y++)
    {
        int pos = y*width + x;
        dev_random_array[pos] = curand_normal(curand_states+x);
    }
}


int main()
{
    const int array_size_width = 1000*1000;
    const int array_size_height = 55;

    //error status
    cudaError_t cuda_status;

    //only chose one GPU
    cuda_status = cudaSetDevice(0);
    float *dev_random_array, *h_random_array;
    curandState *dev_states;
     //allocate memory on the GPU
     //
    float time = 0;
    GPUTimer timer;
    timer.start();

    cuda_status = cudaMalloc((void**)&dev_random_array,sizeof(float)*array_size_width*array_size_height);
    assert(cuda_status == cudaSuccess);
    cuda_status = cudaMalloc((void **)&dev_states,sizeof(curandState)*array_size_width*array_size_height);
    assert(cuda_status == cudaSuccess);
    long clock_for_rand = clock();

    dim3 threads(1024,1);
    dim3 grid((array_size_width+threads.x-1)/threads.x,1);  
    
    kernel_set_random<<<grid,threads>>>(dev_states,array_size_width,array_size_height);//,clock_for_rand);
    kernel_random<<<grid,threads>>>(dev_random_array,array_size_width,array_size_height,dev_states);
    cudaDeviceSynchronize();
    time = timer.seconds();
    printf("time : %f\n", time);
    

    printf("--------------------------------------------\n");
    timer.start();
    curandGenerator_t gen;
    float *dev_random_array2;
    cuda_status = cudaMalloc((void**)&dev_random_array2,sizeof(float)*array_size_width*array_size_height);
    assert(cuda_status == cudaSuccess);

    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT));
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
                1234ULL));

    CURAND_CALL(curandGenerateNormal(gen, dev_random_array2, array_size_width*array_size_height, 0, 1));

    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);
    time = timer.seconds();
    printf("time2 : %f\n", time);

    cuda_status = cudaHostAlloc((void**)&h_random_array, sizeof(float)*array_size_width*array_size_height, 0);
    assert(cuda_status == cudaSuccess);
    // for(int i = 0 ; i<array_size_width*array_size_height ; i++){
    //     h_random_array[i] = 0 ;
    // }

    cuda_status = cudaMemcpyAsync(h_random_array, dev_random_array2, sizeof(float)*array_size_width*array_size_height, 
        cudaMemcpyHostToDevice, 0);
    cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);
    for(int i = 0 ; i<100*5 ; i++){
        std::cout<< h_random_array[i] << " ";
    }
    
    //free
    cudaFreeHost(h_random_array);
    cudaFree(dev_random_array);
    cudaFree(dev_random_array2);
    cudaFree(dev_states);
    return 0;
}