#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "network.h"
#include "layers/layer.h"
#include "layers/implementations/fullyConnectedLayerCPU.cu"
#include <fstream>
#include <string>
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {
    using Common::PerformanceTimer;

	float GraphConvolutionNetworkCPU::loss(float *label, float *predicted) {
		return 0;
	}

	GraphConvolutionNetworkGPU::GraphConvolutionNetworkGPU(int inputDim, int numHiddenLayers, int *hiddenDim, int outputDim, int batchDim) {

	}

	void GraphConvolutionNetworkGPU::forward(float *input, float *output, bool test) {

	}

	void GraphConvolutionNetworkGPU::backward(float *label, float *predicted, float learningRate) {

	}
}
