#include "cuda_fft.h"
#include <iostream>
using std::cout;
using std::endl;


void fft_cuda(thCdouble* h_A, int n, direction dir)
{
	// define device data
	thCdouble *d_A;
	cudaMalloc((thCdouble**) &d_A, n*sizeof(thCdouble));
	cudaMemcpy(d_A, h_A, n*sizeof(thCdouble), cudaMemcpyHostToDevice);

	// invoke kernel
    int threadsPerBlock = MIN(n/2,MAX_THREADS);
    int nBlocks = (n/2-1)/threadsPerBlock + 1;

	// block level kernel call
    fft_kernel_shared<<<nBlocks,threadsPerBlock,2*threadsPerBlock*sizeof(thCdouble)>>>(d_A, n, dir);
    cudaThreadSynchronize();
	cout << endl;


	// continue FFT in global memory
	if(nBlocks>1)
		for(int m=4*MAX_THREADS; m<=n; m<<=1)
		{
			cout << "Going again with m = " << m << endl;
			fft_kernel_finish<<<nBlocks, threadsPerBlock>>>(d_A, m, dir);
			cudaThreadSynchronize();
		}


	// copy back to host
    cudaMemcpy(h_A, d_A, n*sizeof(thCdouble), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
}



void fft_cuda(thCdouble* h_A, int n)
{
	fft_cuda(h_A, n, FORWARD);
}



void ifft_cuda(thCdouble* h_A, int n)
{
	fft_cuda(h_A, n, REVERSE);
}


int main()
{
	// define host data
	int n = 1<<3;
	thCdouble *h_A = new thCdouble[n];
	for(int i=0; i<n; i++)
		h_A[i] = i+1;	

	// call fft
	fft_cuda(h_A, n);

	// print result
	for(int i=0; i<n; i++)
		cout << h_A[i] << endl;

	// memory cleanup
	delete[] h_A;
}
