#pragma once

#include "../common.h"
#include "layer.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class GraphConvolutionLayer : public Layer {
		protected : 
			/* 
				Weight matrix
			*/
			float *W = NULL;
			/* 
				Derivative w.r.t. weight matrix
			*/
			float *dW = NULL;

			/* 
				Input
			*/
			float *A = NULL;
			/* 
				Derivative w.r.t. input
			*/
			float *dA = NULL;
			/* 
				Output of this layer
			*/
			float *Z = NULL;
			

			int inputDim;
			int outputDim;

			int batchDim;
			int numPoints;
			int numFilters;

			bool lastLayer;

	public:
		GraphConvolutionLayer() {};
		GraphConvolutionLayer(int numPoints, int inputDim, int outputDim, int batchDim, int numFilters, bool lastLayer) {
			this->numPoints = numPoints;
			this->inputDim = inputDim;
			this->outputDim = outputDim;
			this->batchDim = batchDim;
			this->numFilters = numFilters;
			this->lastLayer = lastLayer;
		}

		int getInputDim() {
			return inputDim;
		}

		int getOutputDim() {
			return outputDim;
		}
		
		std::vector<float*> forward(std::vector<float*> input, bool test = false);
		void backward(float *incomingGradient, float *outgoingGradient, float learningRate);
	};
}
