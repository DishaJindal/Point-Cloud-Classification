#pragma once


#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/matrix.h"
#include "../utilities/utils.h"
#include "../utilities/parameters.h"
#include "layer.h"
#include "dropoutLayer.h"
#include <fstream>
#include <string>
#include <random>
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

__host__ __device__ inline unsigned int utilhash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

__global__ void dropoutForward(float *input, float *output, float *probabilities, int n, float keepProb, int r) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	thrust::minstd_rand rng = thrust::minstd_rand(utilhash((1 << 31) | r) ^ utilhash(index));
	thrust::uniform_real_distribution<float> dist(0, 1);
	if (index < n) {
		probabilities[index] = (dist(rng) <= 1 - keepProb) ? 1 / keepProb : 0;
		output[index] = input[index] * probabilities[index];
	}
}

__global__ void dropoutBackward(float* input, float* ZDropBackward, float *output, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		output[index] = input[index] * ZDropBackward[index];
	}
}

namespace PointCloudClassification {
	/*
		inputArg -> batchDim x inputDim
		outputArg -> batchDim x inputDim
	*/
	std::vector<float*> DropoutLayerGPU::forward(std::vector<float*> inputArg, bool test) {
		//std::vector<float*> output;
		for (int b = 0; b < batchDim; b++) {
			
			dim3 fullBlocksPerGrid((inputDim * numPoints + blockSize - 1) / blockSize);
			dropoutForward << <fullBlocksPerGrid, blockSize >> > (inputArg[b], this->output[b], this->nodesKeep[b], inputDim * numPoints, Parameters::keep_prob, this->rand_generator());
			//output.push_back(flattened_current_output);
		}
		return output;
}

	std::vector<float*> DropoutLayerGPU::backward(std::vector<float*> incomingGradient, float learningRate) {
		//std::vector<float*> outgoingGradient;
		for (int b = 0; b < batchDim; b++) {
			
			
			dim3 fullBlocksPerGrid((inputDim * numPoints + blockSize - 1) / blockSize);
			dropoutBackward << <fullBlocksPerGrid, blockSize >> > (incomingGradient[b], this->nodesKeep[b], this->outgoing_gradient[b], inputDim * numPoints);
			//outgoingGradient.push_back(flattened_outgoing_gradient);
		}
		return outgoing_gradient;
	}
};