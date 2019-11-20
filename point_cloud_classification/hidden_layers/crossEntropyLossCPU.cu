#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/matrix.h"
#include "../utilities/utils.h"
#include "layer.h"
#include "crossEntropyLoss.h"
#include <fstream>
#include <string>
#include <math.h>

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	class CrossEntropyLossCPU : public CrossEntropyLoss {
	public:
		CrossEntropyLossCPU() {};

		CrossEntropyLossCPU(int batchDim, int numClasses) : CrossEntropyLoss(batchDim, numClasses) {

		}
		
		float cost(std::vector<float*> prediction, std::vector<float*> trueLabel) {
			float sum = 0;
			for (int i = 0; i < batchDim; i++) {
				for (int j = 0; j < numClasses; j++) {
					sum += (-1 * (trueLabel[i][j] * log(prediction[i][j])));
				}
			}
			
			return (sum / (batchDim * numClasses));
		}

		void dcost(float *prediction, float *trueLabel, float *gradient, int batchDim, int numClasses) {
			MatrixCPU* m = new MatrixCPU();
			m->subtract(trueLabel, prediction, batchDim, numClasses, gradient);
		}
	};
}
