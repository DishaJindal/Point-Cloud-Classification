#pragma once

#include "common.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class Layer {
	public:
		//virtual ~Layer() = 0;
		virtual void forward(std::vector<float*> input, std::vector<float*> output, bool test = false) = 0;
		virtual void backward(float *incomingGradient, float *outgoingGradient, float learningRate) = 0;
		virtual int getInputDim() = 0;
		virtual int getOutputDim() = 0;
	};
}
