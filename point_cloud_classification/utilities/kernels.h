#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

namespace Utilities {
	__global__ void kernCrossEntropyLoss(int n, float *predicted, float *label, float *lossForEachLabel);
	__global__ void kernSubtractMatrices(float *input1, float *input2, float *output, int m, int n);
	__global__ void kernMultiplyMatrices(float *input, float *weight, float *output, int m, int n, int k);
	__global__ void kernMultMatricesHammard(float *input1, float *input2, float *output, int m, int n);
	__global__ void kernMultMatricesWithScalar(float *input, float *output, int m, int n, float scalar);
	__global__ void kernTransposeMatrices(float *input, float *output, int m, int n);
	__global__ void kernActivateReLU(float *input, int n);
	__global__ void kernActivateReLUDerivative(float *input, float *output, int n);
	__global__ void kernActivateSoftmax(float *input, int n, int outputDim, float *softmaxDenominator);
}