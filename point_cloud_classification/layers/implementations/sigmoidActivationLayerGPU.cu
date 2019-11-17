#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "../../utilities/kernels.h"
#include "../layer.h"
#include "../sigmoidActivationLayer.h"
#include <fstream>
#include <string>
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	void genArray(int n, float *a) {
		srand(11);

		for (int i = 0; i < n; i++) {
			a[i] = ((2 *((rand() * 1.0 )/ RAND_MAX)) - 1) * 0.0002;
		}
	}

	class sigmoidActivationLayerGPU : public sigmoidActivationLayer {
		sigmoidActivationLayerGPU() {};

		sigmoidActivationLayerGPU(int inputDim, int outputDim, int batchDim, bool lastLayer) {
			sigmoidActivationLayer(inputDim, outputDim, batchDim, lastLayer);
		}

		void forward(float *inputArg, float *outputArg, bool test) {

		}

		void backward(float *incomingGradient, float *outgoingGradient, float learningRate) {

		}
	};
}
