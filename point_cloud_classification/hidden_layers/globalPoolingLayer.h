#pragma once

#include "../common.h"
#include "layer.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class GlobalPoolingLayer : public Layer {
		protected : 
			/* 
				Input
			*/
			std::vector<float*> *Z = NULL;
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
		GlobalPoolingLayer() {};
		GlobalPoolingLayer(int numPoints, int inputDim, int batchDim, bool lastLayer) {
			this->numPoints = numPoints;
			this->inputDim = inputDim;
			this->batchDim = batchDim;
			this->lastLayer = lastLayer;
			this->m = new MatrixCPU();
		}

		int getInputDim() {
			return inputDim;
		}

		int getOutputDim() {
			return inputDim;
		}
		
		std::vector<float*> forward(std::vector<float*> input, bool test = false) = 0;
		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate) = 0;
	};
}
