template <int BLOCK_SIZE> __global__ void
matrixMul(float* C, float* A, float* B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;

    int aEnd   = aBegin + wA - 1;

    int aStep  = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;

    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();


        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

int main(void)
{
    dim3 grid(300, 300);
    dim3 threads(32, 32);
    dim3 dimsA(300*32, 300*32, 1);
    dim3 dimsB(300*32, 300*32, 1);
    float* A, *B, *C;
    
    matrixMul<32><<<grid, threads>>>(C, A, B, 0, 0);
}
