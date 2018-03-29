/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// CUDA Runtime
#include <cuda_runtime.h>
#include <time.h>
// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
#include "slate.h"
#include "quasirandomGenerator_common.h"

void timespec_diff(struct timespec *start, struct timespec *stop,
                   struct timespec *result)
{
    if ((stop->tv_nsec - start->tv_nsec) < 0) {
        result->tv_sec = stop->tv_sec - start->tv_sec - 1;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
    } else {
        result->tv_sec = stop->tv_sec - start->tv_sec;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec;
    }

    return;
}
////////////////////////////////////////////////////////////////////////////////
// CPU code
////////////////////////////////////////////////////////////////////////////////
extern "C" void initQuasirandomGenerator(
    unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION]
);

extern "C" float getQuasirandomValue(
    unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION],
    int i,
    int dim
);

extern "C" double getQuasirandomValue63(INT64 i, int dim);
extern "C" double MoroInvCNDcpu(unsigned int p);

////////////////////////////////////////////////////////////////////////////////
// GPU code
////////////////////////////////////////////////////////////////////////////////
//extern "C" void initTableGPU(unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION]);
//extern "C" void quasirandomGeneratorGPU(float *d_Output, unsigned int seed, unsigned int N);
//extern "C" void inverseCNDgpu(float *d_Output, unsigned int *d_Input, unsigned int N);

const int N = 128 * 1048576;

const char* path = "benchmarks/quasirandomGenerator2/quasirandomGenerator_kernel.cu"; 

int main(int argc, char **argv)
{
    // Start logs
    printf("%s Starting...\n\n", argv[0]);

    unsigned int* tableCPU; 
    slateMalloc((void**)&tableCPU, QRNG_DIMENSIONS * QRNG_RESOLUTION * sizeof(unsigned int));//[QRNG_DIMENSIONS][QRNG_RESOLUTION];

    float *h_OutputGPU, *d_Output;

    int dim, pos;
    double delta, ref, sumDelta, sumRef, L1norm, gpuTime;

    StopWatchInterface *hTimer = NULL;

    if (sizeof(INT64) != 8)
    {
        printf("sizeof(INT64) != 8\n");
        return 0;
    }

    cudaDeviceProp deviceProp;
    int dev = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    if (((deviceProp.major << 4) + deviceProp.minor) < 0x20)
    {
        fprintf(stderr, "quasirandomGenerator requires Compute Capability of SM 2.0 or higher to run.\n");
        exit(EXIT_WAIVED);
    }

    sdkCreateTimer(&hTimer);

    printf("Allocating GPU memory...\n");
   // checkCudaErrors(cudaMalloc((void **)&d_Output, QRNG_DIMENSIONS * N * sizeof(float)));
//(float *)malloc(QRNG_DIMENSIONS * N * sizeof(float));

    printf("Allocating CPU memory...\n");
    slateMalloc((void **)&h_OutputGPU, QRNG_DIMENSIONS * N * sizeof(float));

    printf("Initializing QRNG tables...\n\n");
    initQuasirandomGenerator((unsigned int (*)[QRNG_RESOLUTION]) tableCPU);

    //initTableGPU(tableCPU);
    slateCpyHtoD(tableCPU);

    printf("Testing QRNG...\n\n");
    slateMemset(h_OutputGPU, 0, QRNG_DIMENSIONS * N * sizeof(float));
    int numIterations = 1000; //33; //1000; //9000;

    dim3 threads(128, QRNG_DIMENSIONS);
    slateLaunchKernel(path, "quasirandomGeneratorKernel", N/(threads.x * threads.y), threads, 0, h_OutputGPU, 0, N, tableCPU);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
struct timespec start;
struct timespec end;
struct timespec dif;
clock_gettime(CLOCK_MONOTONIC, &start);

    //slateLaunchBatchKernel(numIterations, path, "quasirandomGeneratorKernel", 128, threads, 0, h_OutputGPU, 0, N, tableCPU);
    for (int i = -1; i < numIterations; i++)
    {
        if (i == 0)
        {
            slateSync();
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }
        dim3 threads(128, QRNG_DIMENSIONS);
        slateLaunchKernel(path, "quasirandomGeneratorKernel", N/(threads.x * threads.y), threads, 0, h_OutputGPU, 0, N, tableCPU);
        //quasirandomGeneratorGPU(h_OutputGPU, 0, N);
    }

    slateSync();

clock_gettime(CLOCK_MONOTONIC, &end);
timespec_diff(&start, &end, &dif);
printf("ktime   %lf\n", (dif.tv_sec + dif.tv_nsec/1e9));

    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer)/(double)numIterations*1e-3;
    printf("quasirandomGenerator, Throughput = %.4f GNumbers/s, Time = %.5f s, Size = %u Numbers, NumDevsUsed = %u, Workgroup = %u\n",
           (double)QRNG_DIMENSIONS * (double)N * 1.0E-9 / gpuTime, gpuTime, QRNG_DIMENSIONS*N, 1, 128*QRNG_DIMENSIONS);

    printf("\nReading GPU results...\n");
    slateCpyDtoH(h_OutputGPU);
/*
    printf("Comparing to the CPU results...\n\n");
    sumDelta = 0;
    sumRef = 0;

    for (dim = 0; dim < QRNG_DIMENSIONS; dim++)
        for (pos = 0; pos < N; pos++)
        {
            ref       = getQuasirandomValue63(pos, dim);
            delta     = (double)h_OutputGPU[dim * N + pos] - ref;
            sumDelta += fabs(delta);
            sumRef   += fabs(ref);
        }

    printf("L1 norm: %E\n", sumDelta / sumRef);
*/
/*
    printf("\nTesting inverseCNDgpu()...\n\n");
    slateMemset(h_OutputGPU, 0, QRNG_DIMENSIONS * N * sizeof(float));

    for (int i = -1; i < numIterations; i++)
    {
        if (i == 0)
        {
            //checkCudaErrors(cudaDeviceSynchronize());
            slateSync();
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }
        slateLaunchKernel(path, "inverseCNDKernel", 128, 128, 0, h_OutputGPU, NULL, QRNG_DIMENSIONS * N);
        //inverseCNDgpu(d_Output, NULL, QRNG_DIMENSIONS * N);
    }
    slateSync();
*/
    sdkStopTimer(&hTimer);
/*
    gpuTime = sdkGetTimerValue(&hTimer)/(double)numIterations*1e-3;
    printf("quasirandomGenerator-inverse, Throughput = %.4f GNumbers/s, Time = %.5f s, Size = %u Numbers, NumDevsUsed = %u, Workgroup = %u\n",
           (double)QRNG_DIMENSIONS * (double)N * 1E-9 / gpuTime, gpuTime, QRNG_DIMENSIONS*N, 1, 128);

    printf("Reading GPU results...\n");
    slateCpyDtoH(h_OutputGPU);

    printf("\nComparing to the CPU results...\n");
    sumDelta = 0;
    sumRef = 0;
    unsigned int distance = ((unsigned int)-1) / (QRNG_DIMENSIONS * N + 1);

    for (pos = 0; pos < QRNG_DIMENSIONS * N; pos++)
    {
        unsigned int d = (pos + 1) * distance;
        ref       = MoroInvCNDcpu(d);
        delta     = (double)h_OutputGPU[pos] - ref;
        sumDelta += fabs(delta);
        sumRef   += fabs(ref);
    }

    printf("L1 norm: %E\n\n", L1norm = sumDelta / sumRef);
*/

    printf("Shutting down...\n");
    sdkDeleteTimer(&hTimer);
    slateHangup();
//    free(h_OutputGPU);
//    checkCudaErrors(cudaFree(d_Output));

    exit(L1norm < 1e-6 ? EXIT_SUCCESS : EXIT_FAILURE);
}
