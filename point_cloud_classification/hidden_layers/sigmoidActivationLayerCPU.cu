#pragma once

#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "layer.h"
#include "sigmoidActivationLayer.h"
#include <fstream>
#include <string>

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	float sigmoid(float z) {
		float sig = (1.0f / (1.0f + exp(-z)));
		return sig;
	}

	/*
		inputArg -> batchDim x inputDim
		outputArg -> batchDim x inputDim
	*/
	std::vector<float*> sigmoidActivationLayerCPU::forward(std::vector<float*> inputArg, bool test) {
		float* flattenedInput = (float*)malloc(batchDim * inputDim * sizeof(float));
		int i = 0;
		for (auto current : inputArg) {
			memcpy(flattenedInput + (i * inputDim), current, inputDim * sizeof(float));
			i++;
		}
		float* flattenedOutput = (float*)malloc(batchDim * outputDim * sizeof(float));

		for(int i = 0; i < batchDim; i++){
			for(int j = 0; j < inputDim; j++){
				flattenedOutput[i * inputDim + j] = sigmoid(flattenedInput[i * inputDim + j]);
			 	/*A[i * inputDim + j] = outputArg[i * inputDim + j];
			 	Z[i * inputDim + j] = inputArg[i * inputDim + j];*/
			}
			}

		std::vector<float*> outputArg;
		for (int i = 0; i < batchDim; i++) {
			outputArg.push_back(flattenedOutput + (i * outputDim));
			A.push_back(flattenedOutput + (i * outputDim));
		}

		return outputArg;
	}

	std::vector<float*> sigmoidActivationLayerCPU::backward(std::vector<float*> incomingGradient, float learningRate) {
		std::vector<float*> outgoingGradient;
		for(int i = 0; i < batchDim; i++){
			float* temp = (float*)malloc(inputDim * sizeof(float));
			for(int j = 0; j < inputDim; j++){
				temp[j] = incomingGradient[i][j] * A[i][j] * (1 - A[i][j]);
			}
			outgoingGradient.push_back(temp);
		}
		return outgoingGradient;
	}
};

