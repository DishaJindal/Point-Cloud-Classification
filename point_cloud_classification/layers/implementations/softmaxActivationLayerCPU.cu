#include "common.h"
#include "../../utilities/kernels.h"
#include "../layer.h"
#include "../softmaxActivationLayer.h"
#include <fstream>
#include <string>

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

	class softmaxActivationLayerCPU : public softmaxActivationLayer {
		softmaxActivationLayerCPU() {};

		softmaxActivationLayerCPU(int inputDim, int outputDim, int batchDim, bool lastLayer) {
			softmaxActivationLayer(inputDim, outputDim, batchDim, lastLayer);
		}

		void forward(float *inputArg, float *outputArg, bool test) {

		}

		void backward(float learningRate, float *incomingGradient, float *outgoingGradient) {

		}
	};
}
