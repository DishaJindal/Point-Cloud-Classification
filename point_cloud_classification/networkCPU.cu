#include "common.h"
#include "network.h"
#include "layers/layer.h"
#include "layers/loss.h"
#include "layers/implementations/fullyConnectedLayerCPU.cu"
#include <fstream>
#include <string>

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {
    using Common::PerformanceTimer;

	GraphConvolutionNetworkCPU::GraphConvolutionNetworkCPU(int inputDim, int outputDim, int batchDim) {

	}

	void GraphConvolutionNetworkCPU::addLayer(Layer* layer){
		this->layers.push_back(layer);
	}

	void GraphConvolutionNetworkCPU::forward(float *input, float *prediction, bool test) {

	}

	void GraphConvolutionNetworkCPU::backward(float *trueLabel, float *prediction, Loss *loss, float learningRate) {

	}
}
