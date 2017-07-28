#include "cuda_fft.h"
#include <stdio.h>

__device__ int d_bitReverse(unsigned int b, int d)
{
    b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
		b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
    b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
    b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
    b = ((b >> 16) | (b << 16));
    return b>>(32-d);
}


// iterative FFT with shared memory
// for n elements, require n/2 threads
// the result of this kernel for n<=2048 is the complete FFT
// for n>2048, fft_kernel_finish must be called to complete
// the FFT in global memory
__global__ void fft_kernel_shared(thCdouble *x, int n, direction dir)
{
 	int tid = threadIdx.x;
    int myId = tid + blockIdx.x * blockDim.x;
    int d = log2f(n);
	n = MIN(n, 2*MAX_THREADS);

    // perform bit-reverse swapping (from global memory to shared)
    // i.e. from global x into block shared array of complexes
    extern __shared__ thCdouble sdata[];
	sdata[ 2*tid ] = x[ d_bitReverse(2*myId, d) ];
	sdata[ 2*tid+1 ] = x[ d_bitReverse(2*myId+1, d) ];
	__syncthreads();

    // perform iterative fft loop
	// m = number of elements at current iteration (tree level)
    for(int m=2; m <= n; m<<=1)
    {
		// set up index variables
    	double v = (2*(dir==REVERSE)-1) * 2 * cuPI / m;
    	int i = tid/(m/2);
   		int j = tid%(m/2);
   	    int k = i*m + j;
		printf("myid = %d, m = %d, i = %d, j = %d, k = %d\n",myId, m, i, j, k);

		// compute value
		thCdouble wj = thrust::polar(1.0, v*j);
       	thCdouble t = sdata[k];
   	    thCdouble u = wj*sdata[k+m/2];
		printf("myId= %d: x[%d] = (%g, %g), x[%d+%d] = (%g, %g), wj = (%g, %g)\n", myId, k, t.real(), t.imag(), k, m/2, (sdata[k+m/2]).real(), (sdata[k+m/2]).imag(), wj.real(), wj.imag());

        sdata[k] = t+u;
        sdata[k+m/2] = t-u;
		printf("myId= %d: NOW x[%d] = (%g, %g), x[%d+%d] = (%g, %g)\n", myId, k, sdata[k].real(), sdata[k].imag(), k, m/2, (sdata[k+m/2]).real(), (sdata[k+m/2]).imag());
        __syncthreads();
    }

    // write result
    x[myId] = sdata[tid];
    x[myId+n/2] = sdata[tid+n/2];
}


// kernel to call after the shared memory kernel
// has performed fft on blocks
// this one will keep going, but working in global memory
// at this point in the computation, each block holds a complete fft
// of the elements it started with. Need to continue fft merging those
// elements until all done
__global__ void fft_kernel_finish(thCdouble *x, int m, direction dir)
{
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockIdx.x * blockDim.x;

	// set up index variables
	double v = (2*(dir==REVERSE)-1) * 2 * cuPI / m;
	int i = myId/(m/2);
	int j = myId%(m/2);
    int k = i*m + j;
	printf("myid = %d, m = %d, i = %d, j = %d, k = %d\n",myId, m, i, j, k);


	// compute value
	thCdouble wj = thrust::polar(1.0, v*j);
   	thCdouble t = x[k];
    thCdouble u = wj*x[k+m/2];
		printf("myId= %d: x[%d] = (%g, %g), x[%d+%d] = (%g, %g), wj = (%g, %g)\n", myId, k, t.real(), t.imag(), k, m/2, (x[k+m/2]).real(), (x[k+m/2]).imag(), wj.real(), wj.imag());
    x[k] = t+u;
    x[k+m/2] = t-u;
		printf("myId= %d: NOW x[%d] = (%g, %g), x[%d+%d] = (%g, %g)\n", myId, k, x[k].real(), x[k].imag(), k, m/2, (x[k+m/2]).real(), (x[k+m/2]).imag());
}
