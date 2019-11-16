#pragma once

#include "common.h"
#include "layer.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class CrossEntropyLoss : public Layer {
		protected : 
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
			float *L = NULL;
			

			int inputDim;
			int batchDim;
			int outputDim;
			bool lastLayer;

		public:
			CrossEntropyLoss() {};
			CrossEntropyLoss(int inputDim, int outputDim, int batchDim, bool lastLayer) {
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
