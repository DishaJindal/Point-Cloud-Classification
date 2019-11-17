#include "common.h"
#include "network.h"
#include "layers/layer.h"
#include "layers/implementations/fullyConnectedLayerCPU.cu"
#include <fstream>
#include <string>

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {
    using Common::PerformanceTimer;

	GraphConvolutionNetworkGPU::GraphConvolutionNetworkCPU(int inputDim, int outputDim, int batchDim) {

	}

	void GraphConvolutionNetworkGPU::addLayer(Layer* layer){
		this->layers.push_back(layer);
	}

	void GraphConvolutionNetworkGPU::forward(float *input, float *prediction, bool test) {

	}

	void GraphConvolutionNetworkGPU::backward(float *trueLabel, float *prediction, Loss *loss, float learningRate) {

	}
}
