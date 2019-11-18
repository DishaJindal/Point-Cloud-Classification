#pragma once
#include "../common.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class Layer {
	public:
		//virtual ~Layer() = 0;
		virtual std::vector<float*> forward(std::vector<float*> input, bool test = false) = 0;
		virtual void backward(float *incomingGradient, float *outgoingGradient, float learningRate) = 0;
		virtual int getInputDim() = 0;
		virtual int getOutputDim() = 0;
	};
}
