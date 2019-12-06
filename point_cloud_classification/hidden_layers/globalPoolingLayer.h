#pragma once

#include "../common.h"
#include "layer.h"
#include "../utilities/matrix.h"

#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class GlobalPoolingLayerCPU : public Layer {
		protected : 
			/* 
				Input
			*/
			std::vector<float*> Z;
			/* 
				Derivative w.r.t. input
			*/
			std::vector<float*> *dZ = NULL;
			/* 
				Output of this layer
			*/
			std::vector<float*> *A = NULL;
			
			int numPoints;
			int inputDim;
			int batchDim;
			bool lastLayer;

			std::vector<float*> mean;
			std::vector<int*> argMax;
			MatrixCPU* m;
			

	public:
		GlobalPoolingLayerCPU() {};
		GlobalPoolingLayerCPU(int numPoints, int inputDim, int batchDim, bool lastLayer) {
			this->numPoints = numPoints;
			this->inputDim = inputDim;
			this->batchDim = batchDim;
			this->lastLayer = lastLayer;
			// Allocate space to save mean required for back propagataion
			for (int i = 0; i < batchDim; i++)
				this->mean.push_back((float*)malloc(inputDim * sizeof(float)));
			// Allocate space to save argmax required for back propagataion
			for (int i = 0; i < batchDim; i++)
				this->argMax.push_back((int*)malloc(inputDim * sizeof(int)));
			this->m = new MatrixCPU();
			
		}

		int getInputDim() {
			return inputDim;
		}

		int getOutputDim() {
			return inputDim;
		}
		
		std::vector<float*> forward(std::vector<float*> input, bool test = false);
		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate);
	};

	class GlobalPoolingLayerGPU : public Layer {
	protected:
		/*
			Input
		*/
		std::vector<float*> Z;
		/*
			Derivative w.r.t. input
		*/
		std::vector<float*> *dZ = NULL;
		/*
			Output of this layer
		*/
		std::vector<float*> *A = NULL;

		int numPoints;
		int inputDim;
		int batchDim;
		bool lastLayer;

		std::vector<float*> mean;
		std::vector<int*> argMax;
		MatrixGPU* m;

		float* current_output;
		std::vector<float*> output;

		float* oneOutgoingGradient;
		std::vector<float*> outgoing_gradient;
		
		

	public:
		GlobalPoolingLayerGPU() {};
		GlobalPoolingLayerGPU(int numPoints, int inputDim, int batchDim, bool lastLayer) {
			this->numPoints = numPoints;
			this->inputDim = inputDim;
			this->batchDim = batchDim;
			this->lastLayer = lastLayer;

			// Allocate space to save mean required for back propagataion
			for (int i = 0; i < batchDim; i++) {
				float* mean_b;
				cudaMalloc((void**)&mean_b, inputDim * sizeof(float));
				this->mean.push_back(mean_b);
			}
			// Allocate space to save argmax required for back propagataion
			for (int i = 0; i < batchDim; i++) {
				int* argmax_b;
				cudaMalloc((void**)&argmax_b, inputDim * sizeof(int));
				this->argMax.push_back(argmax_b);

				cudaMalloc((void**)&current_output, inputDim * 2 * sizeof(float));
				this->output.push_back(current_output);

				cudaMalloc((void**)&oneOutgoingGradient, numPoints * inputDim * sizeof(float));
				outgoing_gradient.push_back(oneOutgoingGradient);
			}
			this->m = new MatrixGPU();

		}

		int getInputDim() {
			return inputDim;
		}

		int getOutputDim() {
			return inputDim;
		}

		std::vector<float*> forward(std::vector<float*> input, bool test = false);
		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate);
	};
}
