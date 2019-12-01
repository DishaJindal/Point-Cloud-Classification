#pragma once

#include "../common.h"
#include "layer.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class softmaxActivationLayerCPU : public Layer {
		protected: 
			/* 
				Input
			*/
			float *Z = NULL;
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
			softmaxActivationLayerCPU() {};
			softmaxActivationLayerCPU(int inputDim, int batchDim, bool lastLayer) {
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

	class softmaxActivationLayerGPU : public Layer {
	protected:
		/*
			Input
		*/
		float *Z = NULL;
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
		softmaxActivationLayerGPU() {};
		softmaxActivationLayerGPU(int inputDim, int batchDim, bool lastLayer) {
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
}

