#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "../utilities/matrix.h"
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
	public:
		CrossEntropyLossGPU() {};

		CrossEntropyLossGPU(int batchDim, int numClasses) : CrossEntropyLoss(batchDim, numClasses) {

		}

		float cost(std::vector<float*> prediction, std::vector<float*> trueLabel) {
			float* flattenedInput;
			cudaMalloc((void**)&flattenedInput, batchDim * numClasses * sizeof(float));
			int i = 0;
			for (auto current : prediction) {
				cudaMemcpy(flattenedInput + (i * numClasses), current, numClasses * sizeof(float), cudaMemcpyDeviceToDevice);
				i++;
			}
			float* softmaxOutput;
			cudaMalloc((void**)&softmaxOutput, batchDim * numClasses * sizeof(float));

			float* temp;
			cudaMalloc((void**)&temp, batchDim * numClasses * sizeof(float));

			MatrixGPU* m = new MatrixGPU();
			m->exp(flattenedInput, batchDim, numClasses, temp);

			float* tempT;
			cudaMalloc((void**)&tempT, batchDim * numClasses * sizeof(float));
			m->transpose(temp, batchDim, numClasses, tempT);
			//cudaFree(temp);

			//m->linearCombination(tempT, tempT, 1000, 0, numClasses, batchDim, tempT);

			float* sum;
			cudaMalloc((void**)&sum, batchDim * numClasses * sizeof(float));
			m->sumAcrossDim1(tempT, numClasses, batchDim, sum); //CHANGE THIS TO SUM --> m->sumAcrossDim1(tempT, outputDim, batchDim, sum);

			//m->linearCombination(sum, sum, 1.0f / 1000, 0, batchDim, 1, sum);

			//dim3 fullBlocksPerGrid((batchDim * inputDim + blockSize - 1) / blockSize);
			m->divide_sum(temp, sum, batchDim, numClasses, softmaxOutput);

			float* all_sum;
			cudaMalloc((void**)&all_sum, batchDim * numClasses * sizeof(float));
			m->sumAcrossDim1(softmaxOutput, batchDim * numClasses, 1, all_sum);

			float* loss = (float*)malloc(sizeof(float));
			cudaMemcpy(loss, all_sum, 1 * sizeof(float), cudaMemcpyDeviceToHost);

			cudaFree(temp);
			cudaFree(tempT);
			cudaFree(sum);

			return loss[0]/(batchDim * numClasses);
			
		}

		std::vector<float*> dcost(std::vector<float*> prediction, std::vector<float*> trueLabel) {
			std::vector<float*> outgoingGradient;
			MatrixGPU* m = new MatrixGPU();

			for (int i = 0; i < batchDim; i++) {
				float* temp;
				cudaMalloc((void**)&temp, numClasses * sizeof(float));
				m->subtract(prediction[i], trueLabel[i], 1, numClasses, temp);
				outgoingGradient.push_back(temp);
			}
			return outgoingGradient;
		}
	};
}
