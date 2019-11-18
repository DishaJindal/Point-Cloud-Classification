#pragma once

#include "common.h"
#include "layer.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class RELUActivationLayer : public Layer {
		protected : 
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
			RELUActivationLayer() {};
			RELUActivationLayer(int inputDim, int outputDim, int batchDim, bool lastLayer) {
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
