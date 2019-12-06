#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "layer.h"
#include "globalPoolingLayer.h"
#include <fstream>
#include <string>
#include "device_launch_parameters.h"

#define blockSize 128

__global__ void kernel_max_pool(float* oneOutgoingGradient, float* incomingGradient, int pts, int idim, int* argm) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i = idx / pts;
	int j = idx % pts;
	// Initialize gradients propagation to all points with 0
	oneOutgoingGradient[idim * j + i] = 0;
	// Update gradient of the max point
	oneOutgoingGradient[idim * argm[i] + i] = incomingGradient[i];
}

__global__ void kernel_variance_pool(float* oneOutgoingGradient, float* incomingGradient, int pts, int idim, float* mea, float* Z) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i = idx / pts;
	int j = idx % pts;
	float del_yj_by_xij = (2 * ((Z[idim * j + i]) - mea[i])) / pts;
	oneOutgoingGradient[idim * j + i] += incomingGradient[i] * del_yj_by_xij;
}

namespace PointCloudClassification {
 
	/*
		inputArg -> N x D (N - number of points per sample)
		outputArg -> 1 x D
		Takes maximum across all points
	*/
	std::vector<float*> GlobalPoolingLayerGPU::forward(std::vector<float*> inputArg, bool test) {
		this->Z = inputArg;

		// Calculate Output
		//std::vector<float*> output;
		for (int b = 0; b < inputArg.size(); b++) {
			

			// Max Pooling and Save Argmax for back prop
			m->maxAcrossDim1(inputArg[b], numPoints, inputDim, this->argMax[b], this->output[b]);

			// Calculate mean and Save it for back prop
			m->meanAcrossDim1(inputArg[b], numPoints, inputDim, this->mean[b]);

			// Variance Pooling
			m->varianceAcrossDim1(inputArg[b], numPoints, inputDim, this->output[b] + inputDim, this->mean[b]);
			//output.push_back(current_output);
		}
		return this->output;
	}


	/*
		incomingGradient -> B*D*2
		returns -> B*N*D

	*/
	std::vector<float*> GlobalPoolingLayerGPU::backward(std::vector<float*> incomingGradient, float learningRate) {
		//std::vector<float*> outgoingGradient;
		for (int b = 0; b < this->batchDim; b++) {
			
			dim3 nBlocks(((inputDim * numPoints) + blockSize - 1) / blockSize);

				
			/* Consume Gradient coming from max pooling
			for (int d = 0; d < this->inputDim; d++) {
				 Initialize gradients propagation to all points with 0
				for (int n = 0; n < this->numPoints; n++) {
					oneOutgoingGradient[this->inputDim * n + d] = 0;
				}
				 Update gradient of the max point
				oneOutgoingGradient[this->inputDim * this->argMax[b][d] + d] = incomingGradient[b][d];
			}

			 Consume Gradient coming from variance pooling
			for (int d = 0; d < this->inputDim; d++) {
				for (int n = 0; n < this->numPoints; n++) {
					float del_yj_by_xij = (2 * ((Z[b][n* inputDim + d]) - this->mean[b][d])) / numPoints;
					oneOutgoingGradient[this->inputDim * n + d] += incomingGradient[b][d] * del_yj_by_xij;
				}
			}*/
			kernel_max_pool<<<nBlocks, blockSize>>> (this->outgoing_gradient[b], incomingGradient[b], this->numPoints, this->inputDim, this->argMax[b]);
			kernel_variance_pool<<<nBlocks, blockSize>>> (this->outgoing_gradient[b], incomingGradient[b], this->numPoints, this->inputDim, this->mean[b], this->Z[b]);
			//outgoingGradient.push_back(oneOutgoingGradient);
		}
		return this->outgoing_gradient;
	}
};
