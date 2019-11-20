#pragma once

#include "../common.h"
#include "loss.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class CrossEntropyLoss : public Loss {
	protected:
		int batchDim;
		int numClasses;
	public:
			CrossEntropyLoss() {};
			CrossEntropyLoss(int batchDim, int numClasses) {
				this->batchDim = batchDim;
				this->numClasses = numClasses;
			}
			
			void dcost(float *prediction, float *trueLabel, float *gradient, int batchDim, int numClasses) = 0;
			float cost(std::vector<float*> prediction, std::vector<float*> trueLabel) = 0;
		};
}
