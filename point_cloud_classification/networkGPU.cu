#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "network.h"
#include "hidden_layers/layer.h"
#include "hidden_layers/fullyConnectedLayerGPU.cu"
#include "hidden_layers/fullyConnectedLayer.h"
#include <fstream>
#include <string>
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {
	GraphConvolutionNetworkGPU::GraphConvolutionNetworkGPU(int inputDim, int numHiddenLayers, int *hiddenDim, int outputDim, int batchDim) {

	}
	void GraphConvolutionNetworkGPU::forward(float *input, float *output, bool test) {

	}
	void GraphConvolutionNetworkGPU::backward(float *output, float *predicted, float learningRate) {

	}
	float GraphConvolutionNetworkGPU::loss(float *label, float *predicted) {
		return 0;
	}
}
