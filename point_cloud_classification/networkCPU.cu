#include "common.h"
#include "network.h"
#include "hidden_layers/layer.h"
#include "hidden_layers/loss.h"
#include "hidden_layers/fullyConnectedLayerCPU.cu"
#include <fstream>
#include <string>

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

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
