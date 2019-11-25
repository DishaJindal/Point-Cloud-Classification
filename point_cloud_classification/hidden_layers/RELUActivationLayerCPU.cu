#pragma once

#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "layer.h"
#include "RELUActivationLayer.h"
#include <fstream>
#include <string>

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	class RELUActivationLayerCPU : public RELUActivationLayer {
	public:
		RELUActivationLayerCPU() {};

		RELUActivationLayerCPU(int inputDim, int batchDim, bool lastLayer) : RELUActivationLayer(inputDim, inputDim, batchDim, lastLayer) {
		
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

			 for(int i = 0; i < batchDim; i++){
			 	for(int j = 0; j < inputDim; j++){
					flattenedOutput[i * inputDim + j] = imax(flattenedInput[i * inputDim + j], 0);
			 	}
			 }
			 //free(flattenedInput);

			 std::vector<float*> outputArg;
			 for (int i = 0; i < batchDim; i++) {
				 outputArg.push_back(flattenedOutput + (i * outputDim));
				 Z.push_back(flattenedInput + (i * inputDim));
			 }
			 //free(flattenedOutput);

			 return outputArg;
		}

		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate) {
			std::vector<float*> outgoingGradient;
			for(int i = 0; i < batchDim; i++){
				float* temp = (float*)malloc(inputDim * sizeof(float));
				for(int j = 0; j < inputDim; j++){
					temp[j] = (Z[i][j] > 0) ? incomingGradient[i][j] : 0;
				}
				outgoingGradient.push_back(temp);
			}
			return outgoingGradient;
		}
	};
}
