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
		float* flattenedInput;
		float* softmaxOutput;
		float* temp;
		float* tempT;
		float* sum;
		float* all_sum;

		float* temp1;
		std::vector<float*> outgoingGradient;
	public:
		CrossEntropyLossGPU() {};

		CrossEntropyLossGPU(int batchDim, int numClasses) : CrossEntropyLoss(batchDim, numClasses) {
			cudaMalloc((void**)&flattenedInput, batchDim * numClasses * sizeof(float));
			cudaMalloc((void**)&softmaxOutput, batchDim * numClasses * sizeof(float));
			cudaMalloc((void**)&temp, batchDim * numClasses * sizeof(float));
			cudaMalloc((void**)&tempT, batchDim * numClasses * sizeof(float));
			cudaMalloc((void**)&sum, batchDim * numClasses * sizeof(float));
			cudaMalloc((void**)&all_sum, batchDim * numClasses * sizeof(float));

			for (int i = 0; i < batchDim; i++) {
				cudaMalloc((void**)&temp1, numClasses * sizeof(float));
				outgoingGradient.push_back(temp1);
			}
		}

		float cost(std::vector<float*> prediction, std::vector<float*> trueLabel) {
			int i = 0;
			for (auto current : prediction) {
				cudaMemcpy(flattenedInput + (i * numClasses), current, numClasses * sizeof(float), cudaMemcpyDeviceToDevice);
				i++;
			}

			MatrixGPU* m = new MatrixGPU();
			m->exp(flattenedInput, batchDim, numClasses, temp);

			m->transpose(temp, batchDim, numClasses, tempT);
			//cudaFree(temp);

			//m->linearCombination(tempT, tempT, 1000, 0, numClasses, batchDim, tempT);

			m->sumAcrossDim1(tempT, numClasses, batchDim, sum); 

			//m->linearCombination(sum, sum, 1.0f / 1000, 0, batchDim, 1, sum);

			//dim3 fullBlocksPerGrid((batchDim * inputDim + blockSize - 1) / blockSize);
			m->divide_sum(temp, sum, batchDim, numClasses, softmaxOutput);

			m->sumAcrossDim1(softmaxOutput, batchDim * numClasses, 1, all_sum);

			float* loss = (float*)malloc(sizeof(float));
			cudaMemcpy(loss, all_sum, 1 * sizeof(float), cudaMemcpyDeviceToHost);

			return loss[0]/(batchDim * numClasses);
		}

		std::vector<float*> dcost(std::vector<float*> prediction, std::vector<float*> trueLabel) {
			
			MatrixGPU* m = new MatrixGPU();

			for (int i = 0; i < batchDim; i++) {
				m->subtract(prediction[i], trueLabel[i], 1, numClasses, outgoingGradient[i]);
			}
			return this->outgoingGradient;
		}
	};
}
