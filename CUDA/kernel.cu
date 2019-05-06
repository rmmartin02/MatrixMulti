/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 128

__global__ void naiveMM(int m, int n, int k, const float *A, const float *B, float* C){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if(id<m*n) 
    {
        int row = (int)(id / m);
        int col = id % n;
        //printf("row %d col %d\n",row,col);
        for(int i = 0; i < k; i++) 
        {
            //printf("row %d col %d k %d i %d n %d A %d B %d %f %f\n",row,col,k,i,n,row*k+i,i*n+col, A[row*k+i],B[i*n+col]);
            sum += A[row * k + i] * B[i * n + col];
            //printf("sum %f\n",sum);
        }
        //printf("%d %d\n",id,sum);
        C[id] = sum;
    }
}

/*
__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    __shared__ float a[TILE_SIZE][TILE_SIZE];
    __shared__ float b[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_SIZE + ty,
       Col = bx * TILE_SIZE + tx;
    float Pvalue = 0;

    for (int i = 0; i < (k-1)/TILE_SIZE+1; ++i) {
        a[ty][tx] = A[Row*k + i*TILE_SIZE+tx];
        b[ty][tx] = B[(i*TILE_SIZE+ty)*n+Col];

       __syncthreads();
       for (int j = 0; j < TILE_SIZE; ++j)
          Pvalue += a[ty][j] * b[j][tx];
       __syncthreads();
    }
    if (Row < m && Col < n)
       C[Row*n+Col] = Pvalue;
}
*/

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE
    int gridSize = (int)ceil((float) (m*n)/BLOCK_SIZE);;

    dim3 dimGrid((n-1)/TILE_SIZE+1, (m-1)/TILE_SIZE+1, 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);



    // Invoke CUDA kernel -----------------------------------------------------
    //<<<num of blocks, num of threads per block

    //INSERT CODE HERE
    //printf("dimGrid1 %d dimGrid2 %d blocksize %d\n",(n-1)/TILE_SIZE+1, (m-1)/TILE_SIZE+1, TILE_SIZE);
    //mysgemm<<<dimGrid,dimBlock>>>(m,n,k,A,B,C);
    //if(which==1){
        //printf("gridSize %d blocksize %d\n",gridSize,BLOCK_SIZE);
    naiveMM<<<gridSize,BLOCK_SIZE>>>(m,n,k,A,B,C);
    //}

}


