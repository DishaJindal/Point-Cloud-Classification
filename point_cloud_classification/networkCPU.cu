#include "common.h"
#include "network.h"
#include "hidden_layers/layer.h"
#include "hidden_layers/loss.h"
#include "graph/graph.h"
#include "utilities/matrix.h"
#include "utilities/parameters.h"
#include "utilities/utils.h"
#include <fstream>
#include <string>

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	NetworkCPU::NetworkCPU(int numClasses, int batchSize) {
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

	void NetworkCPU::buildArchitecture()
	{
		// GCN Layer 1
		PointCloudClassification::GraphConvolutionLayerCPU gcn_layer1(Parameters::num_points, Parameters::input_features, Parameters::gcn1_out_features, Parameters::batch_size, Parameters::chebyshev1_order, false);
		this->gcn_layer1 = gcn_layer1;
		
		// Dropout 1
		PointCloudClassification::DropoutLayerCPU dropout_layer1(Parameters::num_points, Parameters::gcn1_out_features, Parameters::batch_size, false, Parameters::keep_drop_prob1);
		this->dropout_layer1 = dropout_layer1;

		// Global Pooling 1
		PointCloudClassification::GlobalPoolingLayerCPU gp_layer1(Parameters::num_points, Parameters::gcn1_out_features, Parameters::batch_size, false);
		this->gp_layer1 = gp_layer1;

		// GCN Layer 2
		PointCloudClassification::GraphConvolutionLayerCPU gcn_layer2(Parameters::num_points, Parameters::gcn1_out_features, Parameters::gcn2_out_features, Parameters::batch_size, Parameters::chebyshev2_order, false);
		this->gcn_layer2 = gcn_layer2;

		// Dropout 2
		PointCloudClassification::DropoutLayerCPU dropout_layer2(Parameters::num_points, Parameters::gcn2_out_features, Parameters::batch_size, false, Parameters::keep_drop_prob2);
		this->dropout_layer2 = dropout_layer2;

		// Global Pooling 2
		PointCloudClassification::GlobalPoolingLayerCPU gp_layer2(Parameters::num_points, Parameters::gcn2_out_features, Parameters::batch_size, false);
		this->gp_layer2 = gp_layer2;

		// Concatenate GCN Layer 1 and GCN Layer 2


		// Dropout 3
		int cat_features = (Parameters::gcn1_out_features + Parameters::gcn2_out_features);
		PointCloudClassification::DropoutLayerCPU dropout_layer3(cat_features, 2, Parameters::batch_size, false, Parameters::keep_drop_prob3);
		this->dropout_layer3 = dropout_layer3;

		// Fully Connected Layer 1
		PointCloudClassification::FullyConnectedLayerCPU fc_layer1(cat_features * 2, Parameters::fc1_out_features, Parameters::batch_size, false);
		this->fc_layer1 = fc_layer1;

		// ReLU 1
		PointCloudClassification::RELUActivationLayerCPU relu1(Parameters::fc1_out_features, Parameters::batch_size, false);
		this->relu1 = relu1;

		// Dropout 4
		PointCloudClassification::DropoutLayerCPU dropout_layer4(Parameters::fc1_out_features, 1, Parameters::batch_size, false, Parameters::keep_drop_prob4);
		this->dropout_layer4 = dropout_layer4;

		// Fully Connected Layer 2
		PointCloudClassification::FullyConnectedLayerCPU fc_layer2(Parameters::fc1_out_features, Parameters::num_classes, Parameters::batch_size, false);
		this->fc_layer2 = fc_layer2;
	}

	std::vector<float*> NetworkCPU::forward(std::vector<float*> input, bool test) {
		std::vector<float*> output, temp1, temp2;
		output = gcn_layer1.forward(input, false);
		output = dropout_layer1.forward(output, false);
		temp1 = gp_layer1.forward(output, false);

		std::vector<float*> batch_L = std::vector<float*>(input.begin() + Parameters::batch_size, input.end());
		std::vector<float*> output_with_L;
		output_with_L.reserve(output.size() + batch_L.size()); // preallocate memory
		output_with_L.insert(output_with_L.end(), output.begin(), output.end());
		output_with_L.insert(output_with_L.end(), batch_L.begin(), batch_L.end());

		temp2 = gcn_layer2.forward(output_with_L, false);
		temp2 = dropout_layer2.forward(temp2, false);
		temp2 = gp_layer2.forward(temp2, false);

		// Concatenate
		std::vector<float*> cat_vec;
		for (int i = 0; i < Parameters::batch_size; i++) {
			float* cat = (float*)malloc((Parameters::gcn1_out_features + Parameters::gcn2_out_features) * 2 *sizeof(float));
			std::copy(temp1[i], temp1[i] + Parameters::gcn1_out_features * 2, cat);
			std::copy(temp2[i], temp2[i] + Parameters::gcn2_out_features * 2, cat + Parameters::gcn1_out_features * 2);
			cat_vec.push_back(cat);
		}
		
		output = dropout_layer3.forward(cat_vec, false);
		output = fc_layer1.forward(output, false);
		output = relu1.forward(output, false);
		output = dropout_layer4.forward(output, false);
		output = fc_layer2.forward(output, false);
		return output;
	}

	float NetworkCPU::calculateLoss(std::vector<float*> prediction, std::vector<float*> trueLabel) {
		return this->loss->cost(prediction, trueLabel);
	}

	void NetworkCPU::backward(std::vector<float*> prediction, std::vector<float*> trueLabel, float learningRate) {
		// Get the gradient of the loss
		std::vector<float*> dloss = this->loss->dcost(prediction, trueLabel);
		std::vector<float*> incomingGradient(dloss);

		incomingGradient = fc_layer2.backward(incomingGradient, learningRate);
		incomingGradient = dropout_layer4.backward(incomingGradient, learningRate);
		incomingGradient = relu1.backward(incomingGradient, learningRate);
		incomingGradient = fc_layer1.backward(incomingGradient, learningRate);
		incomingGradient = dropout_layer3.backward(incomingGradient, learningRate);

		// Split
		std::vector<float*> gp1, gp2;
		for (int i = 0; i < Parameters::batch_size; i++) {
			gp1.push_back(incomingGradient[i]);
			gp2.push_back(incomingGradient[i] + Parameters::gcn1_out_features * 2);
		}

		gp1 = gp_layer2.backward(gp1, learningRate);
		gp1 = dropout_layer2.backward(gp1, learningRate);
		gp1 = gcn_layer2.backward(gp1, learningRate);

		gp2 = gp_layer1.backward(gp2, learningRate);
		gp2 = dropout_layer1.backward(gp2, learningRate);

		// Add
		for (int i = 0; i < Parameters::batch_size; i++) {
			for (int j = 0; j < Parameters::num_points * Parameters::gcn1_out_features; j++) {
				gp2[i][j] += gp1[i][j];
			}
		}

		gp1 = gcn_layer1.backward(gp2, learningRate);
	}

	void NetworkCPU::train(std::vector<float*> input, std::vector<float*> laplacians, std::vector<float*> label, int n) {

		float* perEpochLoss = (float*)malloc(Parameters::num_epochs * sizeof(float));
		float epochLoss = 0;
		std::vector<float*> classification;// = (std::vector<float*>)malloc(this->batchSize * sizeof(float));
		int num_batches = n / this->batchSize;

		// Iterate for as many epochs..
		for (int ep = 0; ep < Parameters::num_epochs; ep++) {
			epochLoss = 0;

			// Loop batch by batch
			for (int b = 0; b < num_batches; b++) {

				// Grab one batch's data
				std::vector<float*> batch_in = std::vector < float* >(input.begin() + b * this->batchSize, input.begin() + (b + 1) * this->batchSize);
				std::vector<float*> batch_lap = std::vector < float* >(laplacians.begin() + b * this->batchSize, laplacians.begin() + (b + 1) * this->batchSize);
				std::vector<float*> batch;
				batch.reserve(batch_in.size() + batch_lap.size()); // preallocate memory
				batch.insert(batch.end(), batch_in.begin(), batch_in.end());
				batch.insert(batch.end(), batch_lap.begin(), batch_lap.end());
				std::vector<float*> trueLabel = std::vector < float* >(label.begin() + b * this->batchSize, label.begin() + (b + 1) * this->batchSize);
				
				// Forward Pass
				std::vector<float*> prediction = forward(batch, false);

				// Calculate Loss
				float loss = calculateLoss(prediction, trueLabel);
				epochLoss += loss;
				
				// Check Prediction: Can comment this in training later on..
				getClassification(prediction, this->numClasses, classification);
				std::cout << "True Label: ";
				Utilities::printVectorOfFloats(trueLabel, Parameters::num_classes);
				std::cout << std::endl;
				//Utilities::printVector(classification, this->batchSize);
				std::cout << std::endl;
				// Backward Pass
				backward(prediction, trueLabel, Parameters::learning_rate);
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
		std::cout << "Actual Prediction: ";
		Utilities::printVectorOfFloats(prediction, Parameters::num_classes);
		int n = prediction.size();
		PointCloudClassification::softmaxActivationLayerCPU softmaxLayer(numClasses, Parameters::batch_size, false);
		std::vector<float*> pprob = softmaxLayer.forward(prediction, false);
		for (int i = 0; i < n; i++) {
			float maxProb = 0;
			float clazz = 0;
			for (int j = 0; j < classes; j++) {
				if (pprob[i][j] > maxProb) {
					clazz = j;
					maxProb = pprob[i][j];
				}
			}
			classification.push_back(&clazz);
		}
		std::cout << "Prediction: ";
		Utilities::printVectorOfFloats(classification, 1);
	}
}
