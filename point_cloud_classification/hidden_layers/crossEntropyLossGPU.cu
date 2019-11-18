#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "layer.h"
#include "crossEntropyLoss.h"
#include <fstream>
#include <string>

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	class CrossEntropyLossGPU : public CrossEntropyLoss {
		CrossEntropyLossGPU() {};

		float cost(float *prediction, float *trueLabel, int batchDim, int numClasses) {
			return 0;
			
		}

		void dcost(float *prediction, float *trueLabel, float *gradient, int batchDim, int numClasses) {

		}
	};
}
