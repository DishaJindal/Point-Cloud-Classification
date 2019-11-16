#pragma once

#include "common.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
    Common::PerformanceTimer& timer();
	
	class Layer {
	public:
		virtual ~Layer() = 0;
		virtual void forward(float *input, float *output, bool test = false) = 0;
		virtual void backward(float *incomingGradient, float *outgoingGradient, float learningRate) = 0;
		virtual int getInputDim() = 0;
		virtual int getOutputDim() = 0;
	};
}
