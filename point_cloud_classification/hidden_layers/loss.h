#pragma once

#include "../common.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class Loss {
	public:
		//virtual ~Loss() = 0;
		virtual float cost(std::vector<float*> prediction, std::vector<float*> trueLabel) = 0;
		virtual void dcost(float *prediction, float *trueLabel, float *gradient, int batchDim, int numClasses) = 0;
	};
}