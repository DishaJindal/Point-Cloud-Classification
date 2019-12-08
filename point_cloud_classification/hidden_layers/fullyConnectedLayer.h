#pragma once

#include "../common.h"
#include "../utilities/utils.h"
#include "layer.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class FullyConnectedLayerCPU : public Layer {
	protected:
		/*
			Weight matrix - (inputDim x outputDim)
		*/
		float *W = NULL;
		/*
			Bias matrix - (outputDim)
		*/
		float *B = NULL;
		/*
			Derivative w.r.t. weight matrix
		*/
		float *dW = NULL;

		/*
			Input - (batchDim x inputDim)
		*/
		float *A = NULL;
		/*
			Derivative w.r.t. input
		*/
		float *dA = NULL;
		/*
			Output of this layer - (batchDim x outputDim)
		*/
		float *Z = NULL;

		int inputDim;
		
		int outputDim;
		bool lastLayer;

	public:
		int batchDim;
		FullyConnectedLayerCPU() {};
		FullyConnectedLayerCPU(int inputDim, int outputDim, int batchDim, bool lastLayer) {
			this->inputDim = inputDim;
			this->outputDim = outputDim;
			this->batchDim = batchDim;
			this->lastLayer = lastLayer;
			// Randomly initialize weight matrix
			W = (float*)malloc(inputDim * outputDim * sizeof(float));
			B = (float*)malloc(outputDim * sizeof(float));
			dW = (float*)malloc(inputDim * outputDim * sizeof(float));
			A = (float*)malloc(batchDim * inputDim * sizeof(float));
			Utilities::genArray(inputDim * outputDim, W);
			Utilities::genArray(outputDim, B);

		}

		int getInputDim() {
			return inputDim;
		}

		int getOutputDim() {
			return outputDim;
		}
		std::vector<float*> forward(std::vector<float*> input, bool test = false);
		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate);
	};

	class FullyConnectedLayerGPU : public Layer {
	protected:
		/*
			Weight matrix - (inputDim x outputDim)
		*/
		float *W = NULL;
		/*
			Bias matrix - (outputDim)
		*/
		float *B = NULL;
		/*
			Derivative w.r.t. weight matrix
		*/
		float *dW = NULL;

		/*
			Input - (batchDim x inputDim)
		*/
		float *A = NULL;
		/*
			Derivative w.r.t. input
		*/
		float *dA = NULL;
		/*
			Output of this layer - (batchDim x outputDim)
		*/
		float *Z = NULL;

		int inputDim;
		
		int outputDim;
		bool lastLayer;
		float* flattenedInputForward;
		float* flattenedOutputForward;

		float* flattenedInputBackward;
		float* flattenedOutputBackward;

	public:
		int batchDim;
		FullyConnectedLayerGPU() {};
		FullyConnectedLayerGPU(int inputDim, int outputDim, int batchDim, bool lastLayer) {
			this->inputDim = inputDim;
			this->outputDim = outputDim;
			this->batchDim = batchDim;
			this->lastLayer = lastLayer;

			cudaMalloc((void **)&W, inputDim * outputDim * sizeof(float));
			float *weightRand = new float[inputDim * outputDim];
			Utilities::genArray(inputDim * outputDim, weightRand);
			cudaMemcpy(W, weightRand, inputDim * outputDim * sizeof(float), cudaMemcpyHostToDevice);
			cudaMalloc((void **)&B, outputDim * sizeof(float));
			float *biasRand = new float[outputDim];
			Utilities::genArray(outputDim, biasRand);
			cudaMemcpy(B, biasRand, outputDim * sizeof(float), cudaMemcpyHostToDevice);
			cudaMalloc((void**)&A, inputDim * batchDim * sizeof(float));
			cudaMalloc((void**)&dW, inputDim * outputDim * sizeof(float));


			cudaMalloc((void**)&flattenedInputForward, batchDim * inputDim * sizeof(float));
			cudaMalloc((void**)&flattenedOutputForward, batchDim * outputDim * sizeof(float));
			cudaMalloc((void**)&flattenedInputBackward, batchDim * outputDim * sizeof(float));
			cudaMalloc((void**)&flattenedOutputBackward, batchDim * inputDim * sizeof(float));
		}

		int getInputDim() {
			return inputDim;
		}

		int getOutputDim() {
			return outputDim;
		}
		std::vector<float*> forward(std::vector<float*> input, bool test = false);
		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate);
	};
}
