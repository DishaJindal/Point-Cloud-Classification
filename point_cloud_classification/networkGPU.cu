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
#define debug true

namespace PointCloudClassification {

	NetworkGPU::NetworkGPU(int numClasses, int batchSize) {
		this->numClasses = numClasses;
		this->batchSize = batchSize;
	}

	void NetworkGPU::addLayer(Layer* layer) {
		this->layers.push_back(layer);
	}

	void NetworkGPU::setLoss(Loss* loss) {
		this->loss = loss;
	}

	void NetworkGPU::buildArchitecture()
	{
		// GCN Layer 1
		PointCloudClassification::GraphConvolutionLayerGPU gcn_layer1(Parameters::num_points, Parameters::input_features, Parameters::gcn1_out_features, Parameters::batch_size, Parameters::chebyshev1_order, false);
		this->gcn_layer1 = gcn_layer1;

		// Dropout 1
		PointCloudClassification::DropoutLayerGPU dropout_layer1(Parameters::num_points, Parameters::gcn1_out_features, Parameters::batch_size, false, Parameters::keep_drop_prob1);
		this->dropout_layer1 = dropout_layer1;

		// Global Pooling 1
		PointCloudClassification::GlobalPoolingLayerGPU gp_layer1(Parameters::num_points, Parameters::gcn1_out_features, Parameters::batch_size, false);
		this->gp_layer1 = gp_layer1;

		// GCN Layer 2
		PointCloudClassification::GraphConvolutionLayerGPU gcn_layer2(Parameters::num_points, Parameters::gcn1_out_features, Parameters::gcn2_out_features, Parameters::batch_size, Parameters::chebyshev2_order, false);
		this->gcn_layer2 = gcn_layer2;

		// Dropout 2
		PointCloudClassification::DropoutLayerGPU dropout_layer2(Parameters::num_points, Parameters::gcn2_out_features, Parameters::batch_size, false, Parameters::keep_drop_prob2);
		this->dropout_layer2 = dropout_layer2;

		// Global Pooling 2
		PointCloudClassification::GlobalPoolingLayerGPU gp_layer2(Parameters::num_points, Parameters::gcn2_out_features, Parameters::batch_size, false);
		this->gp_layer2 = gp_layer2;

		// Concatenate GCN Layer 1 and GCN Layer 2


		// Dropout 3
		int cat_features = (Parameters::gcn1_out_features + Parameters::gcn2_out_features);
		PointCloudClassification::DropoutLayerGPU dropout_layer3(cat_features, 2, Parameters::batch_size, false, Parameters::keep_drop_prob3);
		this->dropout_layer3 = dropout_layer3;

		// Fully Connected Layer 1
		PointCloudClassification::FullyConnectedLayerGPU fc_layer1(cat_features * 2, Parameters::fc1_out_features, Parameters::batch_size, false);
		this->fc_layer1 = fc_layer1;

		// ReLU 1
		PointCloudClassification::RELUActivationLayerGPU relu1(Parameters::fc1_out_features, Parameters::batch_size, false);
		this->relu1 = relu1;

		// Dropout 4
		PointCloudClassification::DropoutLayerGPU dropout_layer4(Parameters::fc1_out_features, 1, Parameters::batch_size, false, Parameters::keep_drop_prob4);
		this->dropout_layer4 = dropout_layer4;

		// Fully Connected Layer 2
		PointCloudClassification::FullyConnectedLayerGPU fc_layer2(Parameters::fc1_out_features, Parameters::num_classes, Parameters::batch_size, false);
		this->fc_layer2 = fc_layer2;
	}

	std::vector<float*> NetworkGPU::forward(std::vector<float*> input, bool test) {
		std::vector<float*> output, temp1, temp2;
		output = gcn_layer1.forward(input, false);
		if (debug) {
			std::cout << "############################# FORWARD #################################### \n";
			std::cout << "gcn1  " << std::endl;
			Utilities::printVectorOfFloatsGPU(output, 5);
		}
		output = dropout_layer1.forward(output, false);
		if (debug) {
			std::cout << "D 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output, 5);
		}
		temp1 = gp_layer1.forward(output, false);
		if (debug) {
			std::cout << "GP 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(temp1, 5);
		}
		std::vector<float*> batch_L = std::vector<float*>(input.begin() + Parameters::batch_size, input.end());
		std::vector<float*> output_with_L;
		output_with_L.reserve(output.size() + batch_L.size()); // preallocate memory
		output_with_L.insert(output_with_L.end(), output.begin(), output.end());
		output_with_L.insert(output_with_L.end(), batch_L.begin(), batch_L.end());

		temp2 = gcn_layer2.forward(output_with_L, false);
		if (debug) {
			std::cout << "GCN2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(temp2, 5);
		}
		temp2 = dropout_layer2.forward(temp2, false);
		if (debug) {
			std::cout << "D2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(temp2, 5);
		}
		temp2 = gp_layer2.forward(temp2, false);
		if (debug) {
			std::cout << "GP 2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(temp2, 5);
		}
		// Concatenate
		std::vector<float*> cat_vec;
		for (int i = 0; i < Parameters::batch_size; i++) {
			float* cat;
			cudaMalloc((void**)&cat, (Parameters::gcn1_out_features + Parameters::gcn2_out_features) * 2 * sizeof(float));
			cudaMemcpy(cat, temp1[i], (Parameters::gcn1_out_features) * 2 * sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(cat + (Parameters::gcn1_out_features * 2), temp2[i], (Parameters::gcn2_out_features) * 2 * sizeof(float), cudaMemcpyDeviceToDevice);
			cat_vec.push_back(cat);
		}

		if (debug) {
			std::cout << "Cat " << std::endl;
			Utilities::printVectorOfFloatsGPU(cat_vec, 5);
		}
		output = dropout_layer3.forward(cat_vec, false);
		if (debug) {
			std::cout << "D 3 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output, 5);
		}
		output = fc_layer1.forward(output, false);
		if (debug) {
			std::cout << "FC 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output, 5);
		}
		output = relu1.forward(output, false);
		if (debug) {
			std::cout << "R 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output, 5);
		}
		output = dropout_layer4.forward(output, false);
		if (debug) {
			std::cout << "D 4 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output, 5);
		}
		output = fc_layer2.forward(output, false);
		if (debug) {
			std::cout << "FC 2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output, 5);
		}
		return output;
	}

	float NetworkGPU::calculateLoss(std::vector<float*> prediction, std::vector<float*> trueLabel) {
		return this->loss->cost(prediction, trueLabel);
	}

	void NetworkGPU::backward(std::vector<float*> prediction, std::vector<float*> trueLabel, float learningRate) {
		// Get the gradient of the loss
		std::vector<float*> dloss = this->loss->dcost(prediction, trueLabel);
		std::vector<float*> incomingGradient(dloss);

		if (debug) {
			std::cout << "############################# BACKWARD #################################### \n";
			std::cout << "Loss " << std::endl;
			Utilities::printVectorOfFloatsGPU(incomingGradient, 5);
		}
		incomingGradient = fc_layer2.backward(incomingGradient, learningRate);
		if (debug) {
			std::cout << "FC2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(incomingGradient, 5);
		}
		incomingGradient = dropout_layer4.backward(incomingGradient, learningRate);
		if (debug) {
			std::cout << "Dropout 4 " << std::endl;
			Utilities::printVectorOfFloatsGPU(incomingGradient, 5);
		}
		incomingGradient = relu1.backward(incomingGradient, learningRate);
		if (debug) {
			std::cout << "RELU 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(incomingGradient, 5);
		}
		incomingGradient = fc_layer1.backward(incomingGradient, learningRate);
		if (debug) {
			std::cout << "FC1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(incomingGradient, 5);
		}
		incomingGradient = dropout_layer3.backward(incomingGradient, learningRate);
		if (debug) {
			std::cout << "Dropout 3 " << std::endl;
			Utilities::printVectorOfFloatsGPU(incomingGradient, 5);
		}
		// Split
		std::vector<float*> gp1, gp2;
		for (int i = 0; i < Parameters::batch_size; i++) {
			gp1.push_back(incomingGradient[i]);
			gp2.push_back(incomingGradient[i] + Parameters::gcn1_out_features * 2);
		}
		if (debug) {
			std::cout << "Split " << std::endl;
			Utilities::printVectorOfFloatsGPU(incomingGradient, 5);
		}

		gp1 = gp_layer2.backward(gp1, learningRate);
		if (debug) {
			std::cout << "GP 2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(gp1, 5);
		}
		gp1 = dropout_layer2.backward(gp1, learningRate);
		if (debug) {
			std::cout << "Dropout 2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(gp1, 5);
		}
		gp1 = gcn_layer2.backward(gp1, learningRate);
		if (debug) {
			std::cout << "GCN 2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(gp1, 5);
		}

		gp2 = gp_layer1.backward(gp2, learningRate);
		if (debug) {
			std::cout << "GP 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(gp2, 5);
		}
		gp2 = dropout_layer1.backward(gp2, learningRate);
		if (debug) {
			std::cout << "Dropout 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(gp2, 5);
		}
		// Add
		for (int i = 0; i < Parameters::batch_size; i++) {
			MatrixGPU* m = new MatrixGPU();
			m->add(gp1[i], gp2[i], Parameters::num_points, Parameters::gcn1_out_features, gp2[i]);
		}
		if (debug) {
			std::cout << "Add " << std::endl;
			Utilities::printVectorOfFloatsGPU(gp2, 5);
		}
		gp1 = gcn_layer1.backward(gp2, learningRate);
		if (debug) {
			std::cout << "GCN 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(gp1, 5);
		}
	}

	void normalize_data(float* X, int n) {
		// Mean
		float mean[3] = { 0 };
		for (int i = 0; i < n; i++) {
			mean[0] += X[i * 3];
			mean[1] += X[i * 3 + 1];
			mean[2] += X[i * 3 + 2];
		}
		mean[0] = mean[0] / n;
		mean[1] = mean[1] / n;
		mean[2] = mean[2] / n;
		// Mean center
		for (int i = 0; i < n; i++) {
			X[i * 3] -= mean[0];
			X[i * 3 + 1] -= mean[1];
			X[i * 3 + 2] -= mean[2];
		}
		// Furthest distance
		float furthest_dist = 0;
		for (int i = 0; i < n; i++) {
			float dist = 0;
			dist = (X[i * 3] * X[i * 3]) + (X[i * 3 + 1] * X[i * 3 + 1]) + (X[i * 3 + 2] * X[i * 3 + 2]);
			dist = std::sqrt(dist);
			if (dist > furthest_dist) {
				furthest_dist = dist;
			}
		}
		// Divide by furthest distance
		for (int i = 0; i < n; i++) {
			X[i * 3] /= furthest_dist;
			X[i * 3 + 1] /= furthest_dist;
			X[i * 3 + 2] /= furthest_dist;
		}
	}

	std::vector<float*> get_laplacian(std::vector<float*> x_train) {
		std::vector<float*> laplacians;
		for (int i = 0; i < x_train.size(); i++) {
			float* current_sample = x_train[i];
			normalize_data(current_sample, Parameters::num_points);
			Graph::GraphGPU g(current_sample, Parameters::num_points, Parameters::input_features, Parameters::num_neighbours);
			float* L = g.get_Lnorm();
			std::cout << "Constructed graph for " << i << std::endl;
			laplacians.push_back(L);
		}
		return laplacians;
	}

	void NetworkGPU::train(std::vector<float*> input, std::vector<float*> label, int n) {

		float* perEpochLoss;
		cudaMalloc((void**)&perEpochLoss, Parameters::num_epochs * sizeof(float));

		float epochLoss = 0;
		std::vector<float> classification;
		for (int i = 0; i < this->batchSize; i++) {
			classification.push_back(0.0f);
		}
		int num_batches = n / this->batchSize;

		// Iterate for as many epochs..
		for (int ep = 0; ep < Parameters::num_epochs; ep++) {
			std::cout << "****************************Epoch " << ep << "***************************" << std::endl;
			epochLoss = 0;

			// Loop batch by batch
			for (int b = 0; b < num_batches; b++) {

				// Grab one batch's data on CPU
				std::vector<float*> batch_in = std::vector < float* >(input.begin() + b * this->batchSize, input.begin() + (b + 1) * this->batchSize);
				std::vector<float*> trueLabel = std::vector < float* >(label.begin() + b * this->batchSize, label.begin() + (b + 1) * this->batchSize);

				// Prepare Data on GPU
				std::vector<float*> dev_batch;
				dev_batch.reserve(batch_in.size()*2); // preallocate memory
				// Copy batch's data to GPU
				std::vector<float*> dev_in; 
				for (int bi = 0; bi < batch_in.size(); bi++) {
					float* dev_bin;
					cudaMalloc((void**)&dev_bin, Parameters::num_points * Parameters::input_features * sizeof(float));
					cudaMemcpy(dev_bin, batch_in[bi], Parameters::num_points * Parameters::input_features * sizeof(float), cudaMemcpyHostToDevice);
					dev_in.push_back(dev_bin);
				}
				// Laplacian of the batch data
				std::vector<float*> dev_lap = get_laplacian(batch_in);
				//Concatenate Input and Laplacian
				dev_batch.insert(dev_batch.end(), dev_in.begin(), dev_in.end());
				dev_batch.insert(dev_batch.end(), dev_lap.begin(), dev_lap.end());
				// Label on GPU
				std::vector<float*> dev_label;
				for (int bi = 0; bi < batch_in.size(); bi++) {
					float* dev_blab;
					cudaMalloc((void**)&dev_blab, Parameters::num_classes * sizeof(float));
					cudaMemcpy(dev_blab, trueLabel[bi], Parameters::num_classes * sizeof(float), cudaMemcpyHostToDevice);
					dev_label.push_back(dev_blab);
				}

				// Forward Pass
				std::vector<float*> prediction = forward(dev_batch, false);

				// Calculate Loss
				float loss = calculateLoss(prediction, dev_label);
				epochLoss += loss;

				// Check Prediction: Can comment this in training later on..
				getClassification(prediction, this->numClasses, classification);
				std::cout << "True Label: ";
				Utilities::printVectorOfFloatsGPU(dev_label, Parameters::num_classes);
				// Backward Pass
				backward(prediction, dev_label, Parameters::learning_rate);
			}
			epochLoss /= num_batches;
			perEpochLoss[ep] = epochLoss;
			std::cout << "Epoch: " << ep << " Loss: " << epochLoss << "\n";
		}
		std::cout << "Done with training, printing loss\n";
		Utilities::printArray(perEpochLoss, Parameters::num_epochs);
	}

	// Returns classification between [0, classes-1] for each instance
	void NetworkGPU::getClassification(const std::vector<float*> prediction, const int classes, std::vector<float> classification) {
		if (debug) {
			std::cout << "Actual Prediction: ";
			Utilities::printVectorOfFloatsGPU(prediction, Parameters::num_classes);
		}
		int n = prediction.size();
		PointCloudClassification::softmaxActivationLayerGPU softmaxLayer(numClasses, Parameters::batch_size, false);
		std::vector<float*> pprob = softmaxLayer.forward(prediction, false);
		for (int i = 0; i < n; i++) {
			float maxProb = 0;
			float clazz = 0;
			float* pprob_cpu;
			pprob_cpu = (float*)malloc(classes * sizeof(float));

			cudaMemcpy(pprob_cpu, pprob[i], classes * sizeof(float), cudaMemcpyDeviceToHost);
			for (int j = 0; j < classes; j++) {
				if (pprob_cpu[j] > maxProb) {
					clazz = j;
					maxProb = pprob_cpu[j];
				}
			}
			classification[i] = clazz;
		}
		std::cout << "Prediction: ";
		Utilities::printVector(classification, this->batchSize);
	}
	
}
