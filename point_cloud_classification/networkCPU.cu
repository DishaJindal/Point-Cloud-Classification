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

	NetworkCPU::NetworkCPU(int inputFeatures, int numClasses, int batchSize) {
		this->inputFeatures = inputFeatures;
		this->numClasses = numClasses;
		this->batchSize = batchSize;
	}

	void NetworkCPU::addLayer(Layer* layer){
		this->layers.push_back(layer);
	}

	std::vector<float*> NetworkCPU::forward(std::vector<float*> input, bool test) {
		//std::vector<float*> output;
		for(auto layer: this->layers){
			input = layer->forward(input, false);
		}
		return input;
	}

	void NetworkCPU::backward(float *trueLabel, float *prediction, float learningRate) {

	}
}
