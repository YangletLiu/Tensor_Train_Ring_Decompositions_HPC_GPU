#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * 64;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_normal_kernel(curandState *state,
                                int n, 
                                float *result)
{
    int id = threadIdx.x + blockIdx.x * 64;
    float count = 0;
    float2 x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random normals */
    for(int i = 0; i < n/2; i++) {
        x = curand_normal2(&localState);
        /* Check if within one standard deviaton */
        if((x.x > -1.0) && (x.x < 1.0)) {
            count++;
        }
        if((x.y > -1.0) && (x.y < 1.0)) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

int main(int argc, char *argv[])
{

    int i;
    float total;
    curandState *devStates;
    float *devResults, *hostResults;
    int sampleCount = 10000;

    /* Allocate space for results on host */
    hostResults = (float *)calloc(64 * 64, sizeof(float));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&devResults, 64 * 64 * 
              sizeof(float)));

    /* Set results to 0 */
    CUDA_CALL(cudaMemset(devResults, 0, 64 * 64 * 
              sizeof(float)));

    /* Allocate space for prng states on device */
    CUDA_CALL(cudaMalloc((void **)&devStates, 64 * 64 * 
                  sizeof(curandState)));
    
    /* Setup prng states */
    setup_kernel<<<64, 64>>>(devStates);

    /* Generate and use normal pseudo-random  */
    for(i = 0; i < 50; i++) {
            generate_normal_kernel<<<64, 64>>>(devStates, sampleCount, devResults);
    }

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostResults, devResults, 64 * 64 * 
        sizeof(float), cudaMemcpyDeviceToHost));

    /* Show result */
    total = 0;
    for(i = 0; i < 64 * 64; i++) {
        total += hostResults[i];
        std::cout << hostResults[i] << " ";
    }
    printf("\nFraction of normals within 1 standard deviation was %10.13f\n", 
        (float)total / (64.0f * 64.0f * sampleCount * 50.0f));

    /* Cleanup */
    CUDA_CALL(cudaFree(devStates));  
    CUDA_CALL(cudaFree(devResults));
    free(hostResults);
    printf("^^^^ kernel_example PASSED\n");
    return EXIT_SUCCESS;
}

    
    // /* Set results to 0 */
    // CUDA_CALL(cudaMemset(devResults, 0, 64 * 64 * 
    //           sizeof(float)));