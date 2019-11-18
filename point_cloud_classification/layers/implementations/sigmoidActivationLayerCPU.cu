#include "common.h"
#include "../../utilities/kernels.h"
#include "../layer.h"
#include "../sigmoidActivationLayer.h"
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

	class sigmoidActivationLayerCPU : public sigmoidActivationLayer {
	private: float sigmoid(float z) {
			float sig = (1.0f / (1.0f + exp(-z)));
			return sig;
		}
		
		sigmoidActivationLayerCPU() {};

		sigmoidActivationLayerCPU(int inputDim, int outputDim, int batchDim, bool lastLayer) {
			sigmoidActivationLayer(inputDim, outputDim, batchDim, lastLayer);
		}

		

		/*
			inputArg -> batchDim x inputDim
			outputArg -> batchDim x inputDim
		*/
		void forward(float *inputArg, float *outputArg, bool test) {
			for(int i = 0; i < batchDim; i++){
				for(int j = 0; j < inputDim; j++){
					outputArg[i * inputDim + j] = sigmoid(inputArg[i * inputDim + j]);
					A[i * inputDim + j] = outputArg[i * inputDim + j];
					Z[i * inputDim + j] = inputArg[i * inputDim + j];
				}
			}
		}

		void backward(float *incomingGradient, float *outgoingGradient, float learningRate) {
			for(int i = 0; i < batchDim; i++){
				for(int j = 0; j < inputDim; j++){
					outgoingGradient[i * inputDim + j] = incomingGradient[i * inputDim + j] * A[i * inputDim + j] * (1 - A[i * inputDim + j]);
				}
			}
		}
	};
}
