#pragma once

#include "common.h"
#include "loss.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class CrossEntropyLoss : public Loss {
		protected : 

		public:
			CrossEntropyLoss() {};
			
			float cost(float *prediction, float *trueLabel, int batchDim, int numClasses);
			void dcost(float *prediction, float *trueLabel, float *gradient, int batchDim, int numClasses);
		};
}
