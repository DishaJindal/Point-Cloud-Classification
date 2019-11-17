#include "common.h"
#include "../../utilities/kernels.h"
#include "../layer.h"
#include "../fullyConnectedLayer.h"
#include <fstream>
#include <string>
#include "../../utilities/matrixCPU.cu"

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

	class FullyConnectedLayerCPU : public FullyConnectedLayer {
		FullyConnectedLayerCPU() {};

		FullyConnectedLayerCPU(int inputDim, int outputDim, int batchDim, bool lastLayer) {
			FullyConnectedLayer(inputDim, outputDim, batchDim, lastLayer);

			// Randomly initialize weight matrix
			genArray(inputDim * outputDim, W);
		}

		/*
			outputArg = inputArg x W
		*/
		void forward(float *inputArg, float *outputArg, bool test) {
			MatrixCPU* m = new MatrixCPU();
			m->multiply(inputArg, W, batchDim, inputDim, outputDim, outputArg);
		}

		void backward(float *incomingGradient, float *outgoingGradient, float learningRate) {

		}
	};
}
