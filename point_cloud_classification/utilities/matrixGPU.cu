#pragma once
#include "matrix.h"
#include<thrust/device_vector.h>
#include "common.h"
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#ifndef imin
#define imin(a,b) (((a)<(b))?(a):(b))
#endif

#define BLOCK_SIZE 1024

#ifndef cublasSafeCall
#define cublasSafeCall(err)     __cublasSafeCall(err, __FILE__, __LINE__)
#endif

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
	if (CUBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n", __FILE__, __LINE__, err);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

MatrixGPU::MatrixGPU() {
	cublasSafeCall(cublasCreate(&handle));
}

/*
	A -> m x n
	B -> n x p
	output = A x B
*/
__global__ void kernMultiplyMatrices(float *input, float *weight, float *output, int m, int n, int k) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int row = index / k;
	int col = index % k;
	float sum = 0;

	if (col < k && row < m) {
		for (int i = 0; i < n; i++) {
			sum += input[row * n + i] * weight[i*k + col];
		}
		output[row*k + col] = sum;
	}
}

void MatrixGPU::multiply(float* A, float* B, int m, int n, int p, float* output) {
	cublasHandle_t handle;
	cublasSafeCall(cublasCreate(&handle));
	float alpha = 1.0f;
	float beta = 0.0f;
	cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, m, n, &alpha, B, p, A, n, &beta, output, p));
	cublasDestroy(handle);
}


/*
	A -> m x n
	B -> p x n
	output = A x B.T (Multiplies A with the transpose of B)
*/
__global__ void kernMultiplyMatricesTranspose(float *input, float *weight, float *output, int m, int n, int k) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int row = index / k;
	int col = index % k;
	float sum = 0;

	if (col < k && row < m) {
		for (int i = 0; i < n; i++) {
			sum += input[row * n + i] * weight[i + col * n];
		}
		output[row*k + col] = sum;
	}
}
void MatrixGPU::multiplyTranspose(float* A, float* B, int m, int n, int p, float* output) {
	dim3 fullBlocksPerGrid((m * p + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernMultiplyMatricesTranspose <<<fullBlocksPerGrid, BLOCK_SIZE >>> (A, B, output, m, n, p);
	return;


	cublasHandle_t handle;

	cublasSafeCall(cublasCreate(&handle));
	float alpha = 1.0f;
	float beta = 0.0f;
	cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, m, n, &alpha, B, n, A, n, &beta, output, p));
	cublasDestroy(handle);
}
/*
	A -> m x n
	output = A.T
*/
__global__ void kernTransposeMatrices(float *input, float *output, int m, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int row = index / n;
	int col = index % n;

	if (col < n && row < m) {
		int pos = row * n + col;
		int newPos = col * m + row;
		output[newPos] = input[pos];
	}
}
void MatrixGPU::transpose(float* A, int m, int n, float* output) {
	dim3 fullBlocksPerGrid((m * n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernTransposeMatrices << <fullBlocksPerGrid, BLOCK_SIZE >> > (A, output, m, n);
}


/*
	A -> m x n
	B -> m x n
	output = A + B
*/
__global__ void kernAddMatrices(float *input1, float *input2, float *output, int m, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int row = index / n;
	int col = index % n;

	if (col < n && row < m) {
		int pos = row * n + col;
		output[pos] = input1[pos] + input2[pos];
	}

}
void MatrixGPU::add(float* A, float* B, int m, int n, float* output) {
	dim3 fullBlocksPerGrid((m * n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernAddMatrices << <fullBlocksPerGrid, BLOCK_SIZE >> > (A, B, output, m, n);
}


/*
	A -> m x n
	B -> m x n
	output = A - B
*/
__global__ void kernSubtractMatrices(float *input1, float *input2, float *output, int m, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int row = index / n;
	int col = index % n;

	if (col < n && row < m) {
		int pos = row * n + col;
		output[pos] = input1[pos] - input2[pos];
	}

}
void MatrixGPU::subtract(float* A, float* B, int m, int n, float* output) {
	dim3 fullBlocksPerGrid((m * n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernSubtractMatrices << <fullBlocksPerGrid, BLOCK_SIZE >> > (A, B, output, m, n);
}

/*
	A -> m x m
	output = A - I
*/
__global__ void kernSubtractIdentity(float *input1, float *output, int m) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m) {
		int pos = row * m + row;
		output[pos] = input1[pos] - 1;
	}

}
void MatrixGPU::subtractIdentity(float* A, int m) {
	dim3 fullBlocksPerGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernSubtractIdentity << <fullBlocksPerGrid, BLOCK_SIZE >> > (A, A, m);
}

/*
	A -> m x n
	B -> m x n
	output = A - alpha * B
*/
__global__ void kernLCMatrices(float *input1, float *input2, float *output, int m, int n, float alpha, float beta) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int row = index / n;
	int col = index % n;

	if (col < n && row < m) {
		int pos = row * n + col;
		output[pos] = alpha * input1[pos] + beta * input2[pos];
	}

}
void MatrixGPU::subtractWithFactor(float* A, float* B, float alpha, int m, int n, float* output) {
	dim3 fullBlocksPerGrid((m * n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernLCMatrices << <fullBlocksPerGrid, BLOCK_SIZE >> > (A, B, output, m, n, 1, (-1.0f * alpha));
}

/*
	A -> m x n
	B -> m x n
	output = alpha * A + beta * B
*/

void MatrixGPU::linearCombination(float* A, float* B, float alpha, float beta, int m, int n, float* output) {
	dim3 fullBlocksPerGrid((m * n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernLCMatrices << <fullBlocksPerGrid, BLOCK_SIZE >> > (A, B, output, m, n, alpha, beta);
}

__global__ void kernReluForward(float *input, float *output, int m, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < m * n) {
		output[index] = imax(input[index], 0);
	}
}

void MatrixGPU::ReluForward(float* A, int m, int n, float* output) {
	dim3 fullBlocksPerGrid((m * n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernReluForward << <fullBlocksPerGrid, BLOCK_SIZE >> > (A, output, m, n);

}

unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}


template <unsigned int blockSize>
__device__ void kern_warp_reduce_max(volatile float *sdata, unsigned int tid, int width) {
	if (blockSize >= 64) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] = imax(sdata[width * tid + k + width * 32], sdata[width * tid + k]);
		}
	}
	if (blockSize >= 32) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] = imax(sdata[width * tid + k + width * 16], sdata[width * tid + k]);
		}
	}
	if (blockSize >= 16) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] = imax(sdata[width * tid + k + width * 8], sdata[width * tid + k]);
		}
	}
	if (blockSize >= 8) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] = imax(sdata[width * tid + k + width * 4], sdata[width * tid + k]);
		}
	}
	if (blockSize >= 4) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] = imax(sdata[width * tid + k + width * 2], sdata[width * tid + k]);
		}
	};
	if (blockSize >= 2) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] = imax(sdata[width * tid + k + width * 1], sdata[width * tid + k]);
		}
	}
}
template <unsigned int blockSize>
__global__ void kern_reduce_max(float *g_idata, float *g_odata, unsigned int m, unsigned int startColumn, int width, int n) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	for (int k = 0; k < width; ++k) {
		sdata[width * tid + k] = 0;
	}
	while (i < m) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] = imax(g_idata[n * i + k + startColumn] , g_idata[n * i + k + startColumn + n * blockSize]);
		}
		i += gridSize;
	}
	__syncthreads();
	if (blockSize == 1024) {
		if (tid < 512) {
			for (int k = 0; k < width; ++k) {
				sdata[width * tid + k] += sdata[width * tid + k + width * 512];
			}
		} __syncthreads();
	}
	if (blockSize >= 512) {
		if (tid < 256) {
			for (int k = 0; k < width; ++k) {
				sdata[width * tid + k] += sdata[width * tid + k + width * 256];
			}
		} __syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			for (int k = 0; k < width; ++k) {
				sdata[width * tid + k] += sdata[width * tid + k + width * 128];
			}
		} __syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			for (int k = 0; k < width; ++k) {
				sdata[width * tid + k] += sdata[width * tid + k + width * 64];
			}
		} __syncthreads();
	}
	if (tid < 32) kern_warp_reduce_max<blockSize>(sdata, tid, width);
	if (tid == 0) {
		for (int k = 0; k < width; ++k) {
			g_odata[n * blockIdx.x + k + startColumn] = sdata[k];
		}
	}
}
void reduce_max(float* dev_A, int m, int n, float* dev_B) {

	int numFeaturesConsidered = 12;
	int s = m;
	int	threads = (s < 1024 * 2) ? nextPow2((s + 1) / 2) : 1024;
	int	blocks = (s + (threads * 2 - 1)) / (threads * 2);

	if (threads <= 32) {
		numFeaturesConsidered = 6;
	}
	int smemSize = ((threads <= 32) ? 2 : 1) * imin(n, numFeaturesConsidered) * threads * sizeof(float);


	for (int i = 0; i < n; i = i + numFeaturesConsidered) {
		dim3 dimBlock(threads, 1, 1);
		dim3 dimGrid(blocks, 1, 1);
		switch (threads)
		{
		case 1024:
			kern_reduce_max<1024> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
			break;

		case 512:
			kern_reduce_max<512> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
			break;

		case 256:
			kern_reduce_max<256> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
			break;

		case 128:
			kern_reduce_max<128> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
			break;

		case 64:
			kern_reduce_max<64> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
			break;

		case 32:
			kern_reduce_max<32> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
			break;

		case 16:
			kern_reduce_max<16> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
			break;

		case  8:
			kern_reduce_max<8> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
			break;

		case  4:
			kern_reduce_max<4> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
			break;

		case  2:
			kern_reduce_max<2> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
			break;

		case  1:
			kern_reduce_max<1> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
			break;
		}
		cudaDeviceSynchronize();
	}
	s = blocks;
	while (s > 1) {
		float *dev_temp;

		cudaMalloc(&dev_temp, n * s * sizeof(float));
		cudaMemcpy(dev_temp, dev_B, n * s * sizeof(float), cudaMemcpyDeviceToDevice);

		numFeaturesConsidered = 12;
		threads = (s < 1024 * 2) ? nextPow2((s + 1) / 2) : 1024;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);
		if (threads <= 32) {
			numFeaturesConsidered = 6;
		}
		int smemSize = ((threads <= 32) ? 2 : 1) * imin(n, numFeaturesConsidered) * threads * sizeof(float);

		dim3 dimBlock(threads, 1, 1);
		dim3 dimGrid(blocks, 1, 1);

		for (int i = 0; i < n; i = i + numFeaturesConsidered) {
			switch (threads)
			{
			case 1024:
				kern_reduce_max<1024> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
				break;

			case 512:
				kern_reduce_max<512> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
				break;

			case 256:
				kern_reduce_max<256> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
				break;

			case 128:
				kern_reduce_max<128> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
				break;

			case 64:
				kern_reduce_max<64> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
				break;

			case 32:
				kern_reduce_max<32> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
				break;

			case 16:
				kern_reduce_max<16> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
				break;

			case  8:
				kern_reduce_max<8> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
				break;

			case  4:
				kern_reduce_max<4> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
				break;

			case  2:
				kern_reduce_max<2> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
				break;

			case  1:
				kern_reduce_max<1> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n);
				break;
			}
			cudaDeviceSynchronize();
		}
		s = (s + (threads * 2 - 1)) / (threads * 2);
		cudaFree(dev_temp);
	}

}


/*
	A -> m x n
	output = n
*/
void MatrixGPU::maxAcrossDim1(float* A, int m, int n, int* argmaxOutput, float* output) {
	reduce_max(A, m, n, output);
}

template <unsigned int blockSize>
__device__ void kern_warp_reduce_mean(volatile float *sdata, unsigned int tid, int width) {
	if (blockSize >= 64) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] += sdata[width * tid + k + width * 32];
		}
	}
	if (blockSize >= 32) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] += sdata[width * tid + k + width * 16];
		}
	}
	if (blockSize >= 16) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] += sdata[width * tid + k + width * 8];
		}
	}
	if (blockSize >= 8) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] += sdata[width * tid + k + width * 4];
		}
	}
	if (blockSize >= 4) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] += sdata[width * tid + k + width * 2];
		}
	};
	if (blockSize >= 2) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] += sdata[width * tid + k + width * 1];
		}
	}
}
template <unsigned int blockSize>
__global__ void kern_reduce_mean(float *g_idata, float *g_odata, unsigned int m, unsigned int startColumn, int width, int n, float denominator) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	for (int k = 0; k < width; ++k) {
		sdata[width * tid + k] = 0;
	}
	while (i < m) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] += (g_idata[n * i + k + startColumn] + g_idata[n * i + k + startColumn + n * blockSize]) / denominator;
		}
		i += gridSize; 
	}
	__syncthreads();
	if (blockSize == 1024) { if (tid < 512) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] += sdata[width * tid + k + width * 512];
		}
	} __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] += sdata[width * tid + k + width * 256];
		}
	} __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] += sdata[width * tid + k + width * 128];
		}
	} __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) {
		for (int k = 0; k < width; ++k) {
			sdata[width * tid + k] += sdata[width * tid + k + width * 64];
		}
	} __syncthreads(); }
	if (tid < 32) kern_warp_reduce_mean<blockSize>(sdata, tid, width);
	if (tid == 0) {
		for (int k = 0; k < width; ++k) {
			g_odata[n * blockIdx.x + k + startColumn] = sdata[k];
		}
	}
}

void reduce_mean(float* dev_A, int m, int n, float* dev_B, float denominator) {

	int numFeaturesConsidered = 12;
	int s = m;
	int	threads = (s < 1024 * 2) ? nextPow2((s + 1) / 2) : 1024;
	int	blocks = (s + (threads * 2 - 1)) / (threads * 2);

	if (threads <= 32) {
		numFeaturesConsidered = 6;
	}
	int smemSize = ((threads <= 32) ? 2 : 1) * imin(n, numFeaturesConsidered) * threads * sizeof(float);


	for (int i = 0; i < n; i = i + numFeaturesConsidered) {
		dim3 dimBlock(threads, 1, 1);
		dim3 dimGrid(blocks, 1, 1);
		switch (threads)
		{
		case 1024:
			kern_reduce_mean<1024> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, denominator);
			break;

		case 512:
			kern_reduce_mean<512> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, denominator);
			break;

		case 256:
			kern_reduce_mean<256> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, denominator);
			break;

		case 128:
			kern_reduce_mean<128> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, denominator);
			break;

		case 64:
			kern_reduce_mean<64> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, denominator);
			break;

		case 32:
			kern_reduce_mean<32> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, denominator);
			break;

		case 16:
			kern_reduce_mean<16> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, denominator);
			break;

		case  8:
			kern_reduce_mean<8> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, denominator);
			break;

		case  4:
			kern_reduce_mean<4> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, denominator);
			break;

		case  2:
			kern_reduce_mean<2> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, denominator);
			break;

		case  1:
			kern_reduce_mean<1> << < dimGrid, dimBlock, smemSize >> > (dev_A, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, denominator);
			break;
		}
		cudaDeviceSynchronize();
	}
	s = blocks;

	while (s > 1) {
		float *dev_temp;

		cudaMalloc(&dev_temp, n * s * sizeof(float));
		cudaMemcpy(dev_temp, dev_B, n * s * sizeof(float), cudaMemcpyDeviceToDevice);

		numFeaturesConsidered = 12;
		threads = (s < 1024 * 2) ? nextPow2((s + 1) / 2) : 1024;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);
		if (threads <= 32) {
			numFeaturesConsidered = 6;
		}
		int smemSize = ((threads <= 32) ? 2 : 1) * imin(n, numFeaturesConsidered) * threads * sizeof(float);

		dim3 dimBlock(threads, 1, 1);
		dim3 dimGrid(blocks, 1, 1);

		for (int i = 0; i < n; i = i + numFeaturesConsidered) {
			switch (threads)
			{
			case 1024:
				kern_reduce_mean<1024> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, 1);
				break;

			case 512:
				kern_reduce_mean<512> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, 1);
				break;

			case 256:
				kern_reduce_mean<256> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, 1);
				break;

			case 128:
				kern_reduce_mean<128> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, 1);
				break;

			case 64:
				kern_reduce_mean<64> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, 1);
				break;

			case 32:
				kern_reduce_mean<32> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, 1);
				break;

			case 16:
				kern_reduce_mean<16> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, 1);
				break;

			case  8:
				kern_reduce_mean<8> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, 1);
				break;

			case  4:
				kern_reduce_mean<4> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, 1);
				break;

			case  2:
				kern_reduce_mean<2> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, 1);
				break;

			case  1:
				kern_reduce_mean<1> << < dimGrid, dimBlock, smemSize >> > (dev_temp, dev_B, s, i, imin(numFeaturesConsidered, n - i), n, 1);
				break;
			}
			cudaDeviceSynchronize();
		}
		s = (s + (threads * 2 - 1)) / (threads * 2);
		cudaFree(dev_temp);
	}

}
/*
	A -> m x n
	output = n
*/
void MatrixGPU::meanAcrossDim1(float* A, int m, int n, float* output, cudaStream_t stream) {
	thrust::device_vector<float> d_ones(m, 1.0f/m);
	float alpha = 1.0f;
	float beta = 0.0f;
	if (stream != NULL)
		cublasSetStream(handle, stream);
	cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, n, m, &alpha, A, n, thrust::raw_pointer_cast(d_ones.data()), 1, &beta, output,1));
}

void MatrixGPU::sumAcrossDim1(float* A, int m, int n, float* output) {
	linearCombination(A, A, 1000, 0, m, n, A);
	//reduce_mean(A, m, n, output, 1.0f);
	//return;

	cublasHandle_t handle;
	thrust::device_vector<float> d_ones(m, 1.0f);

	cublasSafeCall(cublasCreate(&handle));
	float alpha = 1.0f;
	float beta = 0.0f;
	cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, n, m, &alpha, A, n, thrust::raw_pointer_cast(d_ones.data()), 1, &beta, output, 1));
	cublasDestroy(handle);
	linearCombination(output, output, 1.0f / 1000, 0, m, n, output);
}

__global__ void kernMatrixSubVectorSquare(float *input1, float *input2, float *output, int m, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int row = index / n;
	int col = index % n;

	if (col < n && row < m) {
		int pos = row * n + col;
		float temp = input1[pos] - input2[col];
		output[pos] = (temp) * (temp);
	}

}

/*
	A -> m x n
	output = n
*/
void MatrixGPU::varianceAcrossDim1(float* A, int m, int n, float* output, float* mean, cudaStream_t stream) {
	dim3 fullBlocksPerGrid((m * n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernMatrixSubVectorSquare << <fullBlocksPerGrid, BLOCK_SIZE >> > (A, mean, A, m, n);
	thrust::device_vector<float> d_ones(m, 1.0f / m);
	float alpha = 1.0f;
	float beta = 0.0f;
	if (stream != NULL)
		cublasSetStream(handle, stream);
	cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, n, m, &alpha, A, n, thrust::raw_pointer_cast(d_ones.data()), 1, &beta, output, 1));
}

/*
	Prints matrix A
*/
void MatrixGPU::printMatrix(float* A, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			printf("%f ", A[i * n + j]);
		}
		printf("\n");
	}
}
__global__ void kernGetIdentity(float *input, int m) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int row = index / m;
	int col = index % m;
	
	if (index < m * m) {
		if (row == col) {
			input[index] = 1;
		}
		else {
			input[index] = 0;
		}
	}
}

/*
	returns Identity matrix of given dimension
*/
void MatrixGPU::getIdentityMatrix(int m, float* A) {
	dim3 fullBlocksPerGrid((m * m + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernGetIdentity << <fullBlocksPerGrid, BLOCK_SIZE >> > (A, m);
}

__global__ void kernExp(float* input, int m, int n, float* output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < m * n) {
		output[index] = exp(input[index]);
	}
}

__global__ void kernDivideSum(float* input, float* sum, int m, int n, float* output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int row = index / n;
	if (index < m * n) {
		output[index] = input[index] / sum[row];
	}
}

__global__ void kernCrossEntropy(float* pred, float* trueLabel, int m, int n, float* output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < m * n) {
		output[index] = -1.0f * trueLabel[index] * log(pred[index]);
	}
}

void MatrixGPU::exp(float* input, int batchDim, int inputDim, float* output) {
	dim3 fullBlocksPerGrid((batchDim * inputDim + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernExp << <fullBlocksPerGrid, BLOCK_SIZE >> > (input, batchDim, inputDim, output);
}

void MatrixGPU::divide_sum(float* input, float* sum, int batchDim, int outputDim, float* output) {
	dim3 fullBlocksPerGrid((batchDim * outputDim + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernDivideSum << <fullBlocksPerGrid, BLOCK_SIZE >> > (input, sum, batchDim, outputDim, output);

}

void MatrixGPU::cross_entropy(float* pred, float* trueLabel, int batchDim, int outputDim, float* output) {
	dim3 fullBlocksPerGrid((batchDim * outputDim + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernCrossEntropy << <fullBlocksPerGrid, BLOCK_SIZE >> > (pred, trueLabel, batchDim, outputDim, output);

}