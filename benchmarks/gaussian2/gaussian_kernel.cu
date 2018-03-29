__global__ void Fan1(float* m_cuda, float* a_cuda, int Size, int t)
{   
	//if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) printf(".");
	//printf("blockIDx.x:%d,threadIdx.x:%d,Size:%d,t:%d,Size-1-t:%d\n",blockIdx.x,threadIdx.x,Size,t,Size-1-t);

	if(threadIdx.x + blockIdx.x * blockDim.x < Size-1-t)
    {
	    *(m_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);
    }
}

__global__ void Fan2(float* m_cuda, float* a_cuda, float* b_cuda,int Size, int j1, int t)
{
	if((threadIdx.x + blockIdx.x * blockDim.x < Size-1-t) && (threadIdx.y + blockIdx.y * blockDim.y < Size-t))
    {
	
        int xidx = blockIdx.x * blockDim.x + threadIdx.x;
        int yidx = blockIdx.y * blockDim.y + threadIdx.y;
        //printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
        
        a_cuda[Size*(xidx+1+t)+(yidx+t)] -= m_cuda[Size*(xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];
        //a_cuda[xidx+1+t][yidx+t] -= m_cuda[xidx+1+t][t] * a_cuda[t][yidx+t];
        if(yidx == 0){
            //printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
            //printf("xidx:%d,yidx:%d\n",xidx,yidx);
            b_cuda[xidx+1+t] -= m_cuda[Size*(xidx+1+t)+(yidx+t)] * b_cuda[t];
        }
    }
}
