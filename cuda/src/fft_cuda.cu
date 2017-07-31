#include "fft_cuda.h"
#include "fft_cuda_kernels.h"
#include <stdio.h>


void fft_cuda_transform(thCdouble* h_A, int n, direction dir)
{
	// define device data
	thCdouble *d_A;
	cudaMalloc((thCdouble**) &d_A, n*sizeof(thCdouble));
	thCdouble *d_B;
	cudaMalloc((thCdouble**) &d_B, n*sizeof(thCdouble));
	cudaMemcpy(d_B, h_A, n*sizeof(thCdouble), cudaMemcpyHostToDevice);

	// invoke kernel
    int threadsPerBlock = MIN(n,MAX_THREADS);
    int nBlocks = (n-1)/threadsPerBlock + 1;

	// block level kernel call
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1<<24);
	bit_reverse_kernel<<<nBlocks,threadsPerBlock>>>(d_A, d_B, n);
    cudaThreadSynchronize();
    fft_kernel_shared<<<nBlocks,threadsPerBlock,threadsPerBlock*sizeof(thCdouble)>>>(d_A, n, dir);
    cudaThreadSynchronize();

	// continue FFT in global memory

	if(nBlocks>1)
		for(int m=2*MAX_THREADS; m<=n; m<<=1)
		{
			fft_kernel_finish<<<nBlocks, threadsPerBlock>>>(d_A, m, dir);
			cudaThreadSynchronize();
		}


	// copy back to host
    cudaMemcpy(h_A, d_A, n*sizeof(thCdouble), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
}



void fft_cuda(thCdouble* h_A, int n)
{
	fft_cuda_transform(h_A, n, FORWARD);
}



void ifft_cuda(thCdouble* h_A, int n)
{
	fft_cuda_transform(h_A, n, REVERSE);
}




