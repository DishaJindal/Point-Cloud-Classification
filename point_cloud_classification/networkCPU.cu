#include "common.h"
#include "network.h"
#include "hidden_layers/layer.h"
#include "hidden_layers/loss.h"
#include "hidden_layers/fullyConnectedLayerCPU.cu"
#include "hidden_layers/softmaxActivationLayerCPU.cu"
#include "utilities/parameters.h"
#include "utilities/utils.h"
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
		PointCloudClassification::softmaxActivationLayerCPU softmaxLayer(numClasses, Parameters::batch_size, false);
		this->softmaxFunction = &softmaxLayer;
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

	void NetworkCPU::train(std::vector<float*> input, std::vector<float*> label, int n) {

		float* perEpochLoss = (float*)malloc(Parameters::num_epochs * sizeof(float));
		float epochLoss = 0;
		std::vector<float*> classification(this->batchSize);// = (std::vector<float*>)malloc(this->batchSize * sizeof(float));
		int num_batches = n / this->batchSize;

		// Iterate for as many epochs..
		for (int ep = 0; ep < Parameters::num_epochs; ep++) {
			epochLoss = 0;

			// Loop batch by batch
			for (int b = 0; b < num_batches; b++) {

				// Grab one batch's data
				std::vector<float*> batch = std::vector < float* >(input.begin() + b, input.begin() + b + 1);
				std::vector<float*> trueLabel = std::vector < float* >(label.begin() + b, label.begin() + b + 1);
				
				// Forward Pass
				std::vector<float*> prediction = forward(batch, false);

				// Calculate Loss
				float loss = calculateLoss(prediction, trueLabel);
				epochLoss += loss;
				
				// Check Prediction: Can comment this in training later on..
				getClassification(prediction, this->numClasses, classification);
				std::cout << "True Label: ";
				Utilities::printVector(trueLabel, this->batchSize);
				std::cout << "Prediction: ";
				Utilities::printVector(classification, this->batchSize);

				// Backward Pass

			}
			epochLoss /= num_batches;
			perEpochLoss[ep] = epochLoss;
			std::cout << "Epoch: " << ep << " Loss: " << epochLoss << "\n";
		}
		std::cout << "Done with training, printing loss\n";
		Utilities::printArray(perEpochLoss, Parameters::num_epochs);

	}

	// Returns classification between [0, classes-1] for each instance
	void NetworkCPU::getClassification(const std::vector<float*> prediction, const int classes, std::vector<float*> classification) {
		int n = prediction.size();
		std::vector<float*> pprob = softmaxFunction->forward(prediction, false);
		for (int i = 0; i < n; i++) {
			float maxProb = 0;
			int clazz = 0;
			for (int j = 0; j < classes; j++) {
				if (pprob[i][j] > maxProb) {
					clazz = j;
					maxProb = pprob[i][j];
				}
			}
			classification[i][0] = clazz;
		}
	}
}
