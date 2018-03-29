////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

// ----------------------------------------------------------------------------------------
// Transpose
//
// This file contains both device and host code for transposing a floating-point
// matrix.  It performs several transpose kernels, which incrementally improve performance
// through coalescing, removing shared memory bank conflicts, and eliminating partition
// camping.  Several of the kernels perform a copy, used to represent the best case
// performance that a transpose can achieve.
//
// Please see the whitepaper in the docs folder of the transpose project for a detailed
// description of this performance study.
// ----------------------------------------------------------------------------------------

// Utilities and system includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <slate.h>
#include <time.h>
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

const char* path = "benchmarks/transpose2/transpose.cu";
const char* sSDKsample = "Transpose";
#define TILE_DIM    16
#define BLOCK_ROWS  16

// This sample assumes that MATRIX_SIZE_X = MATRIX_SIZE_Y
int MATRIX_SIZE_X =  10024; //256;//10024; //10024; //1024;
int MATRIX_SIZE_Y = MATRIX_SIZE_X; //1024;
int MUL_FACTOR    = TILE_DIM;

// This sample assumes that MATRIX_SIZE_X = MATRIX_SIZE_Y

#define FLOOR(a,b) (a-(a%b))

// Compute the tile size necessary to illustrate performance cases for SM20+ hardware

// Number of repetitions used for timing.  Two sets of repetitions are performed:
// 1) over kernel launches and 2) inside the kernel over just the loads and stores

// Compute the tile size necessary to illustrate performance cases for SM20+ hardware
int MAX_TILES = (FLOOR(MATRIX_SIZE_X,512) * FLOOR(MATRIX_SIZE_Y,512)) / (TILE_DIM *TILE_DIM);
#define NUM_REPS  6000 //198 //6000
#include <helper_string.h>    // helper for string parsing
#include <helper_image.h>     // helper for image and data comparison
#include <helper_cuda.h>      // helper for cuda error checking functions


// ---------------------
// host utility routines
// ---------------------

void computeTransposeGold(float *gold, float *idata,
                          const  int size_x, const  int size_y)
{
    for (int y = 0; y < size_y; ++y)
    {
        for (int x = 0; x < size_x; ++x)
        {
            gold[(x * size_y) + y] = idata[(y * size_x) + x];
        }
    }
}


void getParams(int argc, char **argv, cudaDeviceProp &deviceProp, int &size_x, int &size_y, int max_tile_dim)
{
    // set matrix size (if (x,y) dim of matrix is not square, then this will have to be modified
    if (checkCmdLineFlag(argc, (const char **)argv, "dimX"))
    {
        size_x = getCmdLineArgumentInt(argc, (const char **) argv, "dimX");

        if (size_x > max_tile_dim)
        {
            printf("> MatrixSize X = %d is greater than the recommended size = %d\n", size_x, max_tile_dim);
        }
        else
        {
            printf("> MatrixSize X = %d\n", size_x);
        }
    }
    else
    {
        size_x = max_tile_dim;
        size_x = FLOOR(size_x, 512);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "dimY"))
    {
        size_y = getCmdLineArgumentInt(argc, (const char **) argv, "dimY");

        if (size_y > max_tile_dim)
        {
            printf("> MatrixSize Y = %d is greater than the recommended size = %d\n", size_y, max_tile_dim);
        }
        else
        {
            printf("> MatrixSize Y = %d\n", size_y);
        }
    }
    else
    {
        size_y = max_tile_dim;

        // If this is SM12 hardware, we want to round down to a multiple of 512
        if ((deviceProp.major == 1 && deviceProp.minor >= 2) || deviceProp.major > 1)
        {
            size_y = FLOOR(size_y, 512);
        }
        else     // else for SM10,SM11 we round down to a multiple of 384
        {
            size_y = FLOOR(size_y, 384);
        }
    }
}


void
showHelp()
{
    printf("\n%s : Command line options\n", sSDKsample);
    printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
    printf("> The default matrix size can be overridden with these parameters\n");
    printf("\t-dimX=row_dim_size (matrix row    dimensions)\n");
    printf("\t-dimY=col_dim_size (matrix column dimensions)\n");
}


// ----
// main
// ----

int
main(int argc, char **argv)
{
    // Start logs
    printf("%s Starting...\n\n", sSDKsample);

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        showHelp();
        return 0;
    }

    int devID = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProp;

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    // compute the scaling factor (for GPUs with fewer MPs)
    float scale_factor, total_tiles;
    scale_factor = std::max((192.0f / (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * (float)deviceProp.multiProcessorCount)), 1.0f);

    printf("> Device %d: \"%s\"\n", devID, deviceProp.name);
    printf("> SM Capability %d.%d detected:\n", deviceProp.major, deviceProp.minor);

    // Calculate number of tiles we will run for the Matrix Transpose performance tests
    int size_x, size_y, max_matrix_dim, matrix_size_test;

    matrix_size_test = 512;  // we round down max_matrix_dim for this perf test
    total_tiles = (float)MAX_TILES / scale_factor;

    max_matrix_dim = FLOOR((int)(floor(sqrt(total_tiles))* TILE_DIM), matrix_size_test);

    // This is the minimum size allowed
    if (max_matrix_dim == 0)
    {
        max_matrix_dim = matrix_size_test;
    }

    printf("> [%s] has %d MP(s) x %d (Cores/MP) = %d (Cores)\n",
           deviceProp.name, deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

    printf("> Compute performance scaling factor = %4.2f\n", scale_factor);

    // Extract parameters if there are any, command line -dimx and -dimy can override
    // any of these settings
    getParams(argc, argv, deviceProp, size_x, size_y, max_matrix_dim);

    if (size_x != size_y)
    {
        printf("\n[%s] does not support non-square matrices (row_dim_size(%d) != col_dim_size(%d))\nExiting...\n\n", sSDKsample, size_x, size_y);
        exit(EXIT_FAILURE);
    }

    if (size_x%TILE_DIM != 0 || size_y%TILE_DIM != 0)
    {
        printf("[%s] Matrix size must be integral multiple of tile size\nExiting...\n\n", sSDKsample);
        exit(EXIT_FAILURE);
    }

    // kernel pointer and descriptor
    void (*kernel)(float *, float *, int, int);
    const char *kernelName;

    // execution configuration parameters
    dim3 grid(size_x/TILE_DIM, size_y/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS);

    if (grid.x < 1 || grid.y < 1)
    {
        printf("[%s] grid size computation incorrect in test \nExiting...\n\n", sSDKsample);
        exit(EXIT_FAILURE);
    }

    // CUDA events
    cudaEvent_t start, stop;

    // size of memory required to store the matrix
    size_t mem_size = static_cast<size_t>(sizeof(float) * size_x*size_y);

    if (2*mem_size > deviceProp.totalGlobalMem)
    {
        printf("Input matrix size is larger than the available device memory!\n");
        printf("Please choose a smaller size matrix\n");
        exit(EXIT_FAILURE);
    }

    // allocate host memory
    float *h_idata;
    slateMalloc((void**)&h_idata, mem_size); //(float *) malloc(mem_size);
    float *h_odata;
    slateMalloc((void**)&h_odata, mem_size);
    float *transposeGold = (float *) malloc(mem_size);
    float *gold;

    // allocate device memory
    //float *d_idata, *d_odata;
    //checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
    //checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

    // initialize host data
    for (int i = 0; i < (size_x*size_y); ++i)
    {
        h_idata[i] = (float) i;
    }

    // copy host data to device
    slateCpyHtoD(h_idata);
    //checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    // Compute reference transpose solution
    // FIXME
    computeTransposeGold(transposeGold, h_idata, size_x, size_y);

    // print out common data for all kernels
    printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n",
           size_x, size_y, size_x/TILE_DIM, size_y/TILE_DIM, TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_ROWS);

    // initialize events
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    //
    // loop over different kernels
    //

    bool success = true;

    for (int k = 0; k<1; k++)
    {
        int i = 2;
        // set kernel pointer
        switch (i)
        {
            case 0:
                //kernel = &copy;
                kernelName = "simple copy       ";
                break;

            case 1:
                //kernel = &copySharedMem;
                kernelName = "shared memory copy";
                break;

            case 2:
                //kernel = &transposeNaive;
                kernelName = "naive             ";
                break;

            case 3:
                //kernel = &transposeCoalesced;
                kernelName = "coalesced         ";
                break;

            case 4:
                //kernel = &transposeNoBankConflicts;
                kernelName = "optimized         ";
                break;

            case 5:
                //kernel = &transposeCoarseGrained;
                kernelName = "coarse-grained    ";
                break;

            case 6:
                //kernel = &transposeFineGrained;
                kernelName = "fine-grained      ";
                break;

            case 7:
                //kernel = &transposeDiagonal;
                kernelName = "diagonal          ";
                break;
        }

        // set reference solution
        /*
        if (kernel == &copy || kernel == &copySharedMem)
        {
            gold = h_idata;
        }
        else if (kernel == &transposeCoarseGrained || kernel == &transposeFineGrained)
        {
            gold = h_odata;   // fine- and coarse-grained kernels are not full transposes, so bypass check
        }
        else
        {
            */
            gold = transposeGold;
        //}

        // Clear error status
        checkCudaErrors(cudaGetLastError());

        // warmup to avoid timing startup
        slateLaunchKernel(path, "transposeNaive", grid, threads, 0, h_odata, h_idata, size_x, size_y);
        slateSync();    
        //kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y);

        // take measurements for loop over kernel launches
        checkCudaErrors(cudaEventRecord(start, 0));
//new
 //       slateLaunchBatchKernel(NUM_REPS, path, "transposeNaive", grid, threads, 0, h_odata, h_idata, size_x, size_y);
        
struct timespec kstart;
struct timespec end;
struct timespec dif;
clock_gettime(CLOCK_MONOTONIC, &kstart);
        for (int i=0; i < NUM_REPS; i++)
        {
            slateLaunchKernel(path, "transposeNaive", grid, threads, 0, h_odata, h_idata, size_x, size_y);
            // Ensure no launch failure
           // checkCudaErrors(cudaGetLastError());
        }
        

        slateSync();
clock_gettime(CLOCK_MONOTONIC, &end);
timespec_diff(&kstart, &end, &dif);
printf("ktime   %lf\n", (dif.tv_sec + dif.tv_nsec/1e9));

        checkCudaErrors(cudaEventRecord(stop, 0));
        checkCudaErrors(cudaEventSynchronize(stop));
        float kernelTime;
        checkCudaErrors(cudaEventElapsedTime(&kernelTime, start, stop));

        slateCpyDtoH(h_odata);
        slateSync();
        //checkCudaErrors(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));
        // FIXME
       bool res = compareData(gold, h_odata, size_x*size_y, 0.01f, 0.0f);
       //bool res = true;
       /*
        for ( size_t i = 0; i < size_x; i++)
        {
            for (size_t j = 0; j < size_y; j++)
            {
                //if (gold[i * size_x + j] != h_odata[i * size_x + j])
                printf("%d, %d ", (int) gold[i * size_x + j], (int) h_odata[i * size_x + j]);
            }
            printf ("\n");
        }
        */
        if (res == false)
        {
            printf("*** %s kernel FAILED ***\n", kernelName);
            success = false;
        }
        // take measurements for loop inside kernel
        //res = compareData(gold, h_odata, size_x*size_y, 0.01f, 0.0f);

        if (res == false)
        {
            printf("*** %s kernel FAILED ***\n", kernelName);
            success = false;
        }
        // report effective bandwidths
        float kernelBandwidth = 2.0f * 1000.0f * mem_size/(1024*1024*1024)/(kernelTime/NUM_REPS);
        printf("transpose %s, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u fp32 elements, NumDevsUsed = %u, Workgroup = %u\n",
               kernelName,
               kernelBandwidth,
               kernelTime/NUM_REPS,
               (size_x *size_y), 1, TILE_DIM *BLOCK_ROWS);

    }

    // cleanup
    free(transposeGold);
   // cudaFree(d_idata);
   // cudaFree(d_odata);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    slateHangup();

    if (!success)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
