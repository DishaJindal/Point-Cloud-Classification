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
		RELUActivationLayerCPU() {};

		RELUActivationLayerCPU(int inputDim, int outputDim, int batchDim, bool lastLayer) {
			RELUActivationLayer(inputDim, outputDim, batchDim, lastLayer);
		}

		/*
			inputArg -> batchDim x inputDim
			outputArg -> batchDim x inputDim
		*/
		std::vector<float*> forward(std::vector<float*> inputArg, bool test) {
			// for(int i = 0; i < batchDim; i++){
			// 	for(int j = 0; j < inputDim; j++){
			// 		outputArg[i * inputDim + j] = imax(inputArg[i * inputDim + j], 0);
			// 		A[i * inputDim + j] = outputArg[i * inputDim + j];
			// 		Z[i * inputDim + j] = inputArg[i * inputDim + j];
			// 	}
			// }
		}

		void backward(float *incomingGradient, float *outgoingGradient, float learningRate) {
			for(int i = 0; i < batchDim; i++){
				for(int j = 0; j < inputDim; j++){
					outgoingGradient[i * inputDim + j] = (Z[i * inputDim + j] > 0) ? incomingGradient[i * inputDim + j] : 0;
				}
			}
		}
	};
}
