#pragma once

#include "../common.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class Loss {
	public:
		//virtual ~Loss() = 0;
		virtual float cost(std::vector<float*> prediction, std::vector<float*> trueLabel) = 0;
		virtual std::vector<float*> dcost(std::vector<float*> prediction, std::vector<float*> trueLabel) = 0;
	};
}