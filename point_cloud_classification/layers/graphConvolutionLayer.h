#pragma once

#include "common.h"
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
			int batchDim;
			int outputDim;
			bool lastLayer;

	public:
		GraphConvolutionLayer() {};
		GraphConvolutionLayer(int inputDim, int outputDim, int batchDim, bool lastLayer) {
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
		
		void forward(float *input, float *output, bool test = false);
		void backward(float *incomingGradient, float *outgoingGradient, float learningRate);
	};
}
