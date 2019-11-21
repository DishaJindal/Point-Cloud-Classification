#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "layer.h"
#include "RELUActivationLayer.h"
#include <fstream>
#include <string>
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {
	class RELUActivationLayerGPU : public RELUActivationLayer {
		RELUActivationLayerGPU() {};

		RELUActivationLayerGPU(int inputDim, int outputDim, int batchDim, bool lastLayer) : RELUActivationLayer(inputDim, outputDim, batchDim, lastLayer) {

		}

		void forward(float *inputArg, float *outputArg, bool test) {

		}

		void backward(float *incomingGradient, float *outgoingGradient, float learningRate) {

		}
	};
}
