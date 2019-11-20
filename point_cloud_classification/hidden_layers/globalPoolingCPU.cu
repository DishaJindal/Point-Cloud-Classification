#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "layer.h"
#include "globalPoolingLayer.h"
#include <fstream>
#include <string>

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	class GlobalPoolingLayerCPU : public GlobalPoolingLayer {
		GlobalPoolingLayerCPU() {};

	public: 
		GlobalPoolingLayerCPU(int inputDim, int outputDim, int batchDim, bool lastLayer) : GlobalPoolingLayer(inputDim, outputDim, batchDim, lastLayer) {
			
		}

		/*
			inputArg -> N x D (N - number of points per sample)
			outputArg -> 1 x D
			Takes maximum across all points
		*/
		std::vector<float*> forward(std::vector<float*> inputArg, bool test) {

		}

		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate) {


		}
	};
}
