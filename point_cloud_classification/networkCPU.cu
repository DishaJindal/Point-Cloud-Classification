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

	void GraphConvolutionNetworkCPU::forward(std::vector<float*> input, std::vector<float*> prediction, bool test) {
		std::vector<float*> output;
		for(auto layer: this->layers){
			layer->forward(input, output, false);
			input = output;
		}
		prediction = output;

	}

	void GraphConvolutionNetworkCPU::backward(float *trueLabel, float *prediction, float learningRate) {

	}
}
