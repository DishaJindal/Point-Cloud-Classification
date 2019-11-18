#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "layer.h"
#include "softmaxActivationLayer.h"
#include <fstream>
#include <string>
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {
	class softmaxActivationLayerGPU : public softmaxActivationLayer {
		softmaxActivationLayerGPU() {};

		softmaxActivationLayerGPU(int inputDim, int outputDim, int batchDim, bool lastLayer) {
			softmaxActivationLayer(inputDim, outputDim, batchDim, lastLayer);
		}

		void forward(std::vector<float*> inputArg, std::vector<float*> outputArg, bool test) {

		}

		void backward(float *incomingGradient, float *outgoingGradient, float learningRate) {

		}
	};
}
