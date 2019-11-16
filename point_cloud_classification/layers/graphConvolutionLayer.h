#pragma once

#include "common.h"
#include "layer.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
    Common::PerformanceTimer& timer();
	class GraphConvolutionLayer : public Layer {
		protected : 
			float *weight = NULL;
			float *inputs = NULL;
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
		void backward(float learningRate, float *incomingGradient, float *outgoingGradient);
	};
}
