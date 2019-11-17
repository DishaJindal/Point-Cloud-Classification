#include <cuda.h>
#include <cuda_runtime.h>
#include "kernels.h"
#include "common.h"
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

namespace Utilities {

	__global__ void kernCrossEntropyLoss(int n, float *predicted, float *label, float *lossForEachLabel) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index < n) {
			lossForEachLabel[index] = -1 * (label[index] * logf(predicted[index]));
		}
	}

	__global__ void kernSubtractMatrices(float *input1, float *input2, float *output, int m, int n) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int row = index / n;
		int col = index % n;

		if (col < n && row < m) {
			int pos = row * n + col;
			output[pos] = input1[pos] - input2[pos];
		}

	}

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

	__global__ void kernMultMatricesHammard(float *input1, float *input2, float *output, int m, int n) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int row = index / n;
		int col = index % n;

		if (col < n && row < m) {
			output[row*n + col] = input1[row*n + col] * input2[row*n + col];
		}
	}

	__global__ void kernMultMatricesWithScalar(float *input, float *output, int m, int n, float scalar) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int row = index / n;
		int col = index % n;

		if (col < n && row < m) {
			int pos = row * n + col;
			output[pos] = scalar * input[pos];
		}
	}


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

	__global__ void kernActivateReLU(float *input, int n) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < n) {
			input[index] = imax(input[index], 0);
		}
	}

	__global__ void kernActivateReLUDerivative(float *input, float *output, int n) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < n) {
			output[index] = (input[index] > 0) ? 1 : 0;
		}
	}

	__global__ void kernActivateSoftmax(float *input, int n, int outputDim, float *softmaxDenominator) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int batchIndex = index / outputDim;
		if (index < n) {
			input[index] = expf(input[index]) / softmaxDenominator[batchIndex];
		}
	}

}