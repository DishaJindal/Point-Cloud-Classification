#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/parameters.h"
#include "../utilities/utils.h"
#include "../utilities/matrix.h"
#include "layer.h"
#include "fullyConnectedLayer.h"
#include <fstream>
#include <string>
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	std::vector<float*> FullyConnectedLayerGPU::forward(std::vector<float*> inputArg, bool test) {
		int i = 0;
		for (auto current : inputArg) {
			cudaMemcpy(flattenedInputForward + (i * inputDim), current, inputDim * sizeof(float), cudaMemcpyDeviceToDevice);
			i++;
		}
		MatrixGPU* m = new MatrixGPU();
		m->multiply(flattenedInputForward, W, batchDim, inputDim, outputDim, flattenedOutputForward);
		//free(flattenedInput);

		// Store input and output of this layer
		cudaMemcpy(A, flattenedInputForward, batchDim * inputDim * sizeof(float), cudaMemcpyDeviceToDevice);

		std::vector<float*> outputArg;
		for (int i = 0; i < batchDim; i++) {
			m->add((float *)(flattenedOutputForward + (i * outputDim)), B, 1, outputDim, (float *)(flattenedOutputForward + (i * outputDim)));
			outputArg.push_back(flattenedOutputForward + (i * outputDim));
		}
		//free(flattenedOutput);

		return outputArg;
	}

	std::vector<float*> FullyConnectedLayerGPU::backward(std::vector<float*> incomingGradient, float learningRate) {
			
			
		int i = 0;
		for (auto current : incomingGradient) {
			cudaMemcpy(flattenedInputBackward + (i * outputDim), current, outputDim * sizeof(float), cudaMemcpyDeviceToDevice);
			i++;
		}

		MatrixGPU* m = new MatrixGPU();

		// Compute gradient w.r.t weights
		float *ATranspose;
		cudaMalloc((void**)&ATranspose, inputDim * batchDim * sizeof(float));
		m->transpose(A, batchDim, inputDim, ATranspose);
		m->multiply(ATranspose, flattenedInputBackward, inputDim, batchDim, outputDim, dW);
		cudaFree(ATranspose);

		// Compute outgoingGradient (w.r.t. input)
		m->multiplyTranspose(flattenedInputBackward, W, batchDim, outputDim, inputDim, flattenedOutputBackward);

		//Update weight matrix
		//m->subtractWithFactor(W, dW, learningRate, inputDim, outputDim, W);
		m->linearCombination(W, dW, (1.0f - Parameters::lamba_reg),-1.0f*learningRate, inputDim, outputDim, W);

		cudaMalloc((void**)&ATranspose, outputDim * batchDim * sizeof(float));
		m->sumAcrossDim1(flattenedInputBackward, batchDim, outputDim, ATranspose);
		m->linearCombination(B, ATranspose, 1.0f, -1.0f*learningRate, 1, outputDim, B);
		cudaFree(ATranspose);


		std::vector<float*> outgoingGradient;
		for (int i = 0; i < batchDim; i++) {
			outgoingGradient.push_back(flattenedOutputBackward + (i * inputDim));
		}
		//free(flattenedOutput);

		return outgoingGradient;

	}
};

