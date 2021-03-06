#pragma once 
#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "../utilities/matrix.h"
#include "layer.h"
#include "RELUActivationLayer.h"
#include <fstream>
#include <string>
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128


__global__ void reluBackward(float* input, float* Z, int n, float* output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		output[index] = (Z[index] > 0) ? input[index] : 0;
	}
}

namespace PointCloudClassification {
	std::vector<float*> RELUActivationLayerGPU::forward(std::vector<float*> inputArg, bool test) {
		int i = 0;
		for (auto current : inputArg) {
			cudaMemcpy(flattenedInput + (i * inputDim), current, inputDim * sizeof(float), cudaMemcpyDeviceToDevice);
			i++;
		}
		MatrixGPU* m = new MatrixGPU();
		m->ReluForward(flattenedInput, batchDim, inputDim, flattenedOutput);
		std::vector<float*> outputArg;
		for (int i = 0; i < batchDim; i++) {
			outputArg.push_back(flattenedOutput + (i * outputDim));
			Z.push_back(flattenedInput + (i * inputDim));
		}
		return outputArg;
	}

	std::vector<float*> RELUActivationLayerGPU::backward(std::vector<float*> incomingGradient, float learningRate) {
		for (int i = 0; i < batchDim; i++) {
			dim3 fullBlocksPerGrid((inputDim + blockSize - 1) / blockSize);
			reluBackward << <fullBlocksPerGrid, blockSize >> > (incomingGradient[i], Z[i], inputDim, outgoingGradient[i]);
		}
		return outgoingGradient;
	}
};

