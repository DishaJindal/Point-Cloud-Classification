#pragma once

#include "common.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class Loss {
	public:
		virtual ~Loss() = 0;
		virtual float cost(float *prediction, float *trueLabel, int batchDim, int numClasses) = 0;
		virtual void dcost(float *prediction, float *trueLabel, float *gradient, int batchDim, int numClasses) = 0;
	};
}