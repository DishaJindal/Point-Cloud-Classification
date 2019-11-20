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
			
			std::vector<float*> dcost(std::vector<float*> prediction, std::vector<float*> trueLabel) = 0;
			float cost(std::vector<float*> prediction, std::vector<float*> trueLabel) = 0;
		};
}
