#include "common.h"
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
		softmaxActivationLayerCPU() {};

		softmaxActivationLayerCPU(int inputDim, int outputDim, int batchDim, bool lastLayer) {
			softmaxActivationLayer(inputDim, outputDim, batchDim, lastLayer);
		}

		/*
			inputArg -> batchDim x inputDim
			outputArg -> batchDim x inputDim
		*/
		void forward(std::vector<float*> inputArg, std::vector<float*> outputArg, bool test) {
			// float* sum = (float*) malloc(batchDim * sizeof(float));
			// for(int i = 0; i < batchDim; i++){
			// 	sum[i] = 0;
			// 	for(int j = 0; j < inputDim; j++){
			// 		float res = exp(inputArg[i * inputDim + j]);
			// 		sum[i] += res;
			// 		outputArg[i * inputDim + j] = res;
			// 	}
			// }

			// for(int i = 0; i < batchDim; i++){
			// 	for(int j = 0; j < inputDim; j++){
			// 		outputArg[i * inputDim + j] /= sum[i];
			// 		A[i * inputDim + j] = outputArg[i * inputDim + j];
			// 		Z[i * inputDim + j] = inputArg[i * inputDim + j];
			// 	}
			// }

		}

		void backward(float *incomingGradient, float *outgoingGradient, float learningRate) {

		}
	};
}
