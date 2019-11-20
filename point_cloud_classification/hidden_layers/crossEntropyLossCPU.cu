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
			float* all_sum = (float*)malloc(batchDim * sizeof(float));
			std::vector<float*> softmaxOutput;
			for (int i = 0; i < batchDim; i++) {
				all_sum[i] = 0;
				for (int j = 0; j < numClasses; j++) {
					float res = exp(prediction[i][j]);
					all_sum[i] += res;
					softmaxOutput[i][j] = res;
				}
			}

			for (int i = 0; i < batchDim; i++) {
				for (int j = 0; j < numClasses; j++) {
					softmaxOutput[i][j] /= all_sum[i];
				}
			}
			
			float sum = 0;
			for (int i = 0; i < batchDim; i++) {
				for (int j = 0; j < numClasses; j++) {
					sum += (-1 * (trueLabel[i][j] * log(softmaxOutput[i][j])));
				}
			}
			
			return (sum / (batchDim * numClasses));
		}

		std::vector<float*> dcost(std::vector<float*> prediction, std::vector<float*> trueLabel) {
			std::vector<float*> outgoingGradient;
			for (int i = 0; i > batchDim; i++) {
				for (int j = 0; j < numClasses; j++) {
					outgoingGradient[i][j] = trueLabel[i][j] - prediction[i][j];
				}
			}
			return outgoingGradient;
		}
	};
}
