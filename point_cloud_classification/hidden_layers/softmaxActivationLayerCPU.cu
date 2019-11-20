#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "layer.h"
#include "softmaxActivationLayer.h"
#include <fstream>
#include <string>

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {
	class softmaxActivationLayerCPU : public softmaxActivationLayer {
	public:
		softmaxActivationLayerCPU() {};

		softmaxActivationLayerCPU(int inputDim, int batchDim, bool lastLayer) : softmaxActivationLayer(inputDim, inputDim, batchDim, lastLayer) {
		
		}

		/*
			inputArg -> batchDim x inputDim
			outputArg -> batchDim x inputDim
		*/
		std::vector<float*> forward(std::vector<float*> inputArg, bool test) {
			float* flattenedInput = (float*)malloc(batchDim * inputDim * sizeof(float));
			int i = 0;
			for (auto current : inputArg) {
				memcpy(flattenedInput + (i * inputDim), current, inputDim * sizeof(float));
				i++;
			}
			float* flattenedOutput = (float*)malloc(batchDim * outputDim * sizeof(float));

			float* sum = (float*) malloc(batchDim * sizeof(float));
			 for(int i = 0; i < batchDim; i++){
			 	sum[i] = 0;
			 	for(int j = 0; j < inputDim; j++){
			 		float res = exp(flattenedInput[i * inputDim + j]);
			 		sum[i] += res;
					flattenedOutput[i * inputDim + j] = res;
			 	}
			 }

			 for(int i = 0; i < batchDim; i++){
			 	for(int j = 0; j < inputDim; j++){
					flattenedOutput[i * inputDim + j] /= sum[i];
			 		/*A[i * inputDim + j] = outputArg[i * inputDim + j];
			 		Z[i * inputDim + j] = inputArg[i * inputDim + j];*/
			 	}
			 }

			 std::vector<float*> outputArg;
			 for (int i = 0; i < batchDim; i++) {
				 outputArg.push_back(flattenedOutput + (i * outputDim));
			 }

			 return outputArg;
		}

		void backward(float *incomingGradient, float *outgoingGradient, float learningRate) {

		}
	};
}
