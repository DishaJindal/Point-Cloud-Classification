#pragma once

#include "../common.h"
#include "layer.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class sigmoidActivationLayerCPU : public Layer {
		protected: 
			/* 
				Input
			*/
			std::vector<float*> A;
			/* 
				Derivative w.r.t. input
			*/
			float *dZ = NULL;
			
			int inputDim;
			
			int outputDim;
			bool lastLayer;

		public:
			int batchDim;
			sigmoidActivationLayerCPU() {};
			sigmoidActivationLayerCPU(int inputDim, int batchDim, bool lastLayer) {
				this->inputDim = inputDim;
				this->outputDim = inputDim;
				this->batchDim = batchDim;
				this->lastLayer = lastLayer;
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

	class sigmoidActivationLayerGPU : public Layer {
	protected:
		/*
			Input
		*/
		std::vector<float*> A;
		/*
			Derivative w.r.t. input
		*/
		float *dZ = NULL;

		int inputDim;
		int batchDim;
		int outputDim;
		bool lastLayer;

	public:
		sigmoidActivationLayerGPU() {};
		sigmoidActivationLayerGPU(int inputDim, int outputDim, int batchDim, bool lastLayer) {
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

		void forward(float *inputArg, float *outputArg, bool test);
		void backward(float *incomingGradient, float *outgoingGradient, float learningRate);
	};
}

