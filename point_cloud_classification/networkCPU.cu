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

	void NetworkCPU::setLoss(Loss* loss) {
		this->loss = loss;
	}

	std::vector<float*> NetworkCPU::forward(std::vector<float*> input, bool test) {
		//std::vector<float*> output;
		for(auto layer: this->layers){
			input = layer->forward(input, false);
		}
		return input;
	}

	float NetworkCPU::calculateLoss(std::vector<float*> prediction, std::vector<float*> trueLabel) {
		return this->loss->cost(prediction, trueLabel);
	}

	void NetworkCPU::backward(std::vector<float*> prediction, std::vector<float*> trueLabel, float learningRate) {
		// Get the gradient of the loss
		std::vector<float*> dloss = this->loss->dcost(prediction, trueLabel);

		// Loop over all layers in reverse order
		int number_of_layers = this->layers.size();
		std::vector<float*> incomingGradient(dloss);
		
		std::cout << "LOSS GRAD: " << std::endl;
		std::cout << incomingGradient[0][0] << " " << incomingGradient[0][1] << " " << incomingGradient[0][2] << std::endl;
		std::cout << incomingGradient[1][0] << " " << incomingGradient[1][1] << " " << incomingGradient[1][2] << std::endl;
		std::cout << incomingGradient[2][0] << " " << incomingGradient[2][1] << " " << incomingGradient[2][2] << std::endl;
		std::cout << std::endl;

		for (int i = number_of_layers - 1; i >= 0; i--) {
			Layer* layer = this->layers[i];
			incomingGradient = layer->backward(incomingGradient, learningRate);

			std::cout << "INCOMING GRAD: " << std::endl;
			std::cout << incomingGradient[0][0] << " " << incomingGradient[0][1] << " " << incomingGradient[0][2] << std::endl;
			std::cout << incomingGradient[1][0] << " " << incomingGradient[1][1] << " " << incomingGradient[1][2] << std::endl;
			std::cout << incomingGradient[2][0] << " " << incomingGradient[2][1] << " " << incomingGradient[2][2] << std::endl;
			std::cout << std::endl;

		}
	}
}
