#pragma once

#include "../common.h"
#include "layer.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class RELUActivationLayer : public Layer {
	protected:
		/*
			Input
		*/
		std::vector<float *> Z;
		/*
			Derivative w.r.t. input
		*/
		float *dZ = NULL;
		/*
			Output of this layer
		*/
		float *A = NULL;


		int inputDim;
		int batchDim;
		int outputDim;
		bool lastLayer;

	public:
		RELUActivationLayer() {};
		RELUActivationLayer(int inputDim, int outputDim ,int batchDim, bool lastLayer) {
			this->inputDim = inputDim;
			this->outputDim = outputDim;
			this->batchDim = batchDim;
			this->lastLayer = lastLayer;
		}

		int getInputDim() {
			return inputDim;
		}

		int getOutputDim() {
			return outputDim;
		}

		std::vector<float*> forward(std::vector<float*> input, bool test = false) = 0;
		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate) = 0;
	};



	class RELUActivationLayerCPU : public RELUActivationLayer {
	public:
		RELUActivationLayerCPU() {}
		RELUActivationLayerCPU(int inputDim, int batchDim, bool lastLayer) : RELUActivationLayer(inputDim, inputDim, batchDim, lastLayer) {
		}
		std::vector<float*> forward(std::vector<float*> input, bool test = false);
		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate);
	};

	class RELUActivationLayerGPU : public RELUActivationLayer {
	public:
		float* flattenedOutput;
		float* flattenedInput;
		std::vector<float*> outgoingGradient;
		RELUActivationLayerGPU() {}
		RELUActivationLayerGPU(int inputDim, int batchDim, bool lastLayer) : RELUActivationLayer(inputDim, inputDim, batchDim, lastLayer) {
			cudaMalloc((void**)&flattenedInput, batchDim * inputDim * sizeof(float));
			cudaMalloc((void**)&flattenedOutput, batchDim * outputDim * sizeof(float));
			for (int i = 0; i < batchDim; i++) {
				float* temp;
				cudaMalloc((void**)&temp, inputDim * sizeof(float));
				outgoingGradient.push_back(temp);
			}
		}
		std::vector<float*> forward(std::vector<float*> input, bool test = false);
		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate);
	};

}
