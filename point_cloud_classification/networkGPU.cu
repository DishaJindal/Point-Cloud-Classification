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
#include <chrono>

using namespace std::chrono;

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128
#define debug true
#define memStats true
#define time false

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

	void memPrint(const char* statement) {
		if (memStats) {
			size_t free_byte;
			size_t total_byte;
			if (cudaSuccess != cudaMemGetInfo(&free_byte, &total_byte))
			{
				checkCUDAError("Error: cudaMemGetInfo fails");
			}
			double free_db = (double)free_byte;
			double total_db = (double)total_byte;
			double used_db = total_db - free_db;
			printf(statement);
			printf(". GPU memory usage: used = %f, free = %f MB, total = %f MB\n\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
		}
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
		auto start = high_resolution_clock::now();
		output_gn1 = gcn_layer1.forward(input, false);
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Graph Convolution Layer 1 Forward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}
		if (debug) {
			std::cout << "############################# FORWARD #################################### \n";
			std::cout << "gcn1  " << std::endl;
			Utilities::printVectorOfFloatsGPU(output_gn1, 10);
		}

		start = high_resolution_clock::now();
		output_d1 = dropout_layer1.forward(output_gn1, false);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Dropout 1 Forward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}

		if (debug) {
			std::cout << "D 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output_d1, 10);
		}
		start = high_resolution_clock::now();
		output_gp1 = gp_layer1.forward(output_d1, false);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Global Pooling 1 Forward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}
		if (debug) {
			std::cout << "GP 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output_gp1, 10);
		}
		std::vector<float*> batch_L = std::vector<float*>(input.begin() + Parameters::batch_size, input.end());
		std::vector<float*> output_with_L;
		output_with_L.reserve(output_d1.size() + batch_L.size()); // preallocate memory
		output_with_L.insert(output_with_L.end(), output_d1.begin(), output_d1.end());
		output_with_L.insert(output_with_L.end(), batch_L.begin(), batch_L.end());

		start = high_resolution_clock::now();
		output_gcn2 = gcn_layer2.forward(output_with_L, false);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Graph Convolution Layer 2 Forward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}

		if (debug) {
			std::cout << "GCN2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output_gcn2, 10);
		}
		start = high_resolution_clock::now();
		output_d2 = dropout_layer2.forward(output_gcn2, false);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Dropout 2 Forward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}

		if (debug) {
			std::cout << "D2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output_d2, 10);
		}
		start = high_resolution_clock::now();
		output_gp2 = gp_layer2.forward(output_d2, false);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Global Pooling 2 Forward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}

		if (debug) {
			std::cout << "GP 2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output_gp2, 10);
		}
		// Concatenate
		std::vector<float*> cat_vec;
		for (int i = 0; i < Parameters::batch_size; i++) {
			float* cat;
			cudaMalloc((void**)&cat, (Parameters::gcn1_out_features + Parameters::gcn2_out_features) * 2 * sizeof(float));
			cudaMemcpy(cat, output_gp1[i], (Parameters::gcn1_out_features) * 2 * sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(cat + (Parameters::gcn1_out_features * 2), output_gp2[i], (Parameters::gcn2_out_features) * 2 * sizeof(float), cudaMemcpyDeviceToDevice);
			cat_vec.push_back(cat);
		}

		if (debug) {
			std::cout << "Cat " << std::endl;
			Utilities::printVectorOfFloatsGPU(cat_vec, 10);
		}

		start = high_resolution_clock::now();
		output_d3 = dropout_layer3.forward(cat_vec, false);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Dropout 3 Forward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}

		if (debug) {
			std::cout << "D 3 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output_d3, 10);
		}
		start = high_resolution_clock::now();
		output_fc1 = fc_layer1.forward(output_d3, false);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Fully Connected Layer 1 Forward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}

		if (debug) {
			std::cout << "FC 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output_fc1, 10);
		}
		
		start = high_resolution_clock::now();
		output_r1 = relu1.forward(output_fc1, false);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (RELU 1 Forward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}

		if (debug) {
			std::cout << "R 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output_r1, 10);
		}
		

		start = high_resolution_clock::now();
		output_d4 = dropout_layer4.forward(output_r1, false);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Dropout 4 Forward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}

		if (debug) {
			std::cout << "D 4 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output_d4, 10);
		}
		
		start = high_resolution_clock::now();
		output_fc2 = fc_layer2.forward(output_d4, false);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Fully Connected Layer 2 Forward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}

		if (debug) {
			std::cout << "FC 2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(output_fc2, 10);
		}
		
		return output_fc2;
	}

	float NetworkGPU::calculateLoss(std::vector<float*> prediction, std::vector<float*> trueLabel) {
		return this->loss->cost(prediction, trueLabel);
	}

	void NetworkGPU::backward(std::vector<float*> prediction, std::vector<float*> trueLabel, float learningRate) {
		

		// Get the gradient of the loss
		auto start = high_resolution_clock::now();
		std::vector<float*> dloss = this->loss->dcost(prediction, trueLabel);
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Loss gradient / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}
		std::vector<float*> incomingGradient(dloss);

		if (debug) {
			std::cout << "############################# BACKWARD #################################### \n";
			std::cout << "Loss " << std::endl;
			Utilities::printVectorOfFloatsGPU(incomingGradient, 10);
		}
		start = high_resolution_clock::now();
		incomingGradient = fc_layer2.backward(incomingGradient, learningRate);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Fully Connected Layer 2 Backward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}
		if (debug) {
			std::cout << "FC2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(incomingGradient, 10);
		}
		
		start = high_resolution_clock::now();
		incomingGradient = dropout_layer4.backward(incomingGradient, learningRate);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Dropout 4 Backward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}
		if (debug) {
			std::cout << "Dropout 4 " << std::endl;
			Utilities::printVectorOfFloatsGPU(incomingGradient, 10);
		}
		
		start = high_resolution_clock::now();
		incomingGradient = relu1.backward(incomingGradient, learningRate);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (RELU 1 Backward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}
		if (debug) {
			std::cout << "RELU 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(incomingGradient, 10);
		}
		start = high_resolution_clock::now();
		incomingGradient = fc_layer1.backward(incomingGradient, learningRate);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Fully Connected Layer 1 Backward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}
		if (debug) {
			std::cout << "FC1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(incomingGradient, 10);
		}
		start = high_resolution_clock::now();
		incomingGradient = dropout_layer3.backward(incomingGradient, learningRate);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Dropout Layer 3 Backward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}
		if (debug) {
			std::cout << "Dropout 3 " << std::endl;
			Utilities::printVectorOfFloatsGPU(incomingGradient, 10);
		}
		// Split
		std::vector<float*> gp1, gp2;
		for (int i = 0; i < Parameters::batch_size; i++) {
			gp1.push_back(incomingGradient[i]);
			gp2.push_back(incomingGradient[i] + Parameters::gcn1_out_features * 2);
		}
		if (debug) {
			std::cout << "Split " << std::endl;
			Utilities::printVectorOfFloatsGPU(incomingGradient, 10);
		}
		start = high_resolution_clock::now();
		gp1 = gp_layer2.backward(gp1, learningRate);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Global Pooling Layer 2 Backward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}
		if (debug) {
			std::cout << "GP 2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(gp1, 10);
		}
		
		start = high_resolution_clock::now();
		gp1 = dropout_layer2.backward(gp1, learningRate);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Dropout Layer 2 Backward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}
		if (debug) {
			std::cout << "Dropout 2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(gp1, 10);
		}
		
		start = high_resolution_clock::now();
		gp1 = gcn_layer2.backward(gp1, learningRate);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Graph Convolution Layer 2 Backward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}
		if (debug) {
			std::cout << "GCN 2 " << std::endl;
			Utilities::printVectorOfFloatsGPU(gp1, 10);
		}
		
		start = high_resolution_clock::now();
		gp2 = gp_layer1.backward(gp2, learningRate);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Global Pooling Layer 1 Backward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}
		if (debug) {
			std::cout << "GP 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(gp2, 10);
		}
		
		start = high_resolution_clock::now();
		gp2 = dropout_layer1.backward(gp2, learningRate);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Dropout 1 Backward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}
		if (debug) {
			std::cout << "Dropout 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(gp2, 10);
		}
		
		// Add
		for (int i = 0; i < Parameters::batch_size; i++) {
			MatrixGPU* m = new MatrixGPU();
			m->add(gp1[i], gp2[i], Parameters::num_points, Parameters::gcn1_out_features, gp2[i]);
		}
		if (debug) {
			std::cout << "Add " << std::endl;
			Utilities::printVectorOfFloatsGPU(gp2, 10);
		}
		start = high_resolution_clock::now();
		gp1 = gcn_layer1.backward(gp2, learningRate);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		if (time) {
			std::cout << "GPU : (Graph Convolution Layer 1 Backward / batch) ==> " << duration.count() << " microseconds" << std::endl;
		}
		if (debug) {
			std::cout << "GCN 1 " << std::endl;
			Utilities::printVectorOfFloatsGPU(gp1, 10);
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

	void get_laplacian(std::vector<float*> x_train, std::vector<float*> dev_lap) {
		for (int i = 0; i < x_train.size(); i++) {
			Graph::GraphGPU g(x_train[i], dev_lap[i], Parameters::num_points, Parameters::input_features, Parameters::num_neighbours);
			if (debug) {
				std::cout << "Constructed graph for " << i << std::endl;
			}
		}
	}

	void NetworkGPU::freeForwardGarbage() {
		for (int i = 0; i < output_gn1.size(); i++) {
			cudaFree(output_gn1[i]);
		}
		output_gn1.clear();
		for (int i = 0; i < output_d1.size(); i++) {
			cudaFree(output_d1[i]);
		}
		output_d1.clear();
		for (int i = 0; i < output_gp1.size(); i++) {
			cudaFree(output_gp1[i]);
		}
		output_gp1.clear();
		for (int i = 0; i < output_gcn2.size(); i++) {
			cudaFree(output_gcn2[i]);
		}
		output_gcn2.clear();
		for (int i = 0; i < output_d2.size(); i++) {
			cudaFree(output_d2[i]);
		}
		output_d2.clear();
		for (int i = 0; i < output_gp2.size(); i++) {
			cudaFree(output_gp2[i]);
		}
		output_gp2.clear();
		for (int i = 0; i < output_d3.size(); i++) {
			cudaFree(output_d3[i]);
		}
		output_d3.clear();
		for (int i = 0; i < output_fc1.size(); i++) {
			cudaFree(output_fc1[i]);
		}
		output_fc1.clear();
		for (int i = 0; i < output_r1.size(); i++) {
			cudaFree(output_r1[i]);
		}
		output_r1.clear();
		for (int i = 0; i < output_d4.size(); i++) {
			cudaFree(output_d4[i]);
		}
		output_d4.clear();
		for (int i = 0; i < output_fc2.size(); i++) {
			cudaFree(output_fc2[i]);
		}
		output_fc2.clear();
	}

	void NetworkGPU::train(std::vector<float*> input, std::vector<float*> label, int n) {

		float* perEpochLoss = (float*)malloc(Parameters::num_epochs * sizeof(float));
		// Space for Laplacian per batch

		float epochLoss = 0;
		std::vector<float> classification;
		for (int i = 0; i < this->batchSize; i++) {
			classification.push_back(0.0f);
		}
		int num_batches = n / this->batchSize; 
		
		// One batch worth Laplacians
		memPrint("Before Batch Graph Cons");
		std::vector<float*> dev_lap;
		for (int i = 0; i < Parameters::batch_size; i++) {
			float* l;
			cudaMalloc((void**)&l, Parameters::num_points * Parameters::num_points * sizeof(float));
			dev_lap.push_back(l);
		}
		memPrint("After Batch Graph Cons");

		// One batch worth Input Data
		memPrint("Before Input Data");
		std::vector<float*> dev_in;
		for (int bi = 0; bi < Parameters::batch_size; bi++) {
			float* dev_bin;
			cudaMalloc((void**)&dev_bin, Parameters::num_points * Parameters::input_features * sizeof(float));
			dev_in.push_back(dev_bin);
		}
		memPrint("After Input Data");

		// Label on GPU
		memPrint("Before Label");
		std::vector<float*> dev_label;
		for (int bi = 0; bi < Parameters::batch_size; bi++) {
			float* dev_blab;
			cudaMalloc((void**)&dev_blab, Parameters::num_classes * sizeof(float));
			dev_label.push_back(dev_blab);
		}
		memPrint("After Label");

		// Prepare Forward Input
		memPrint("Before Forward Input");
		std::vector<float*> dev_batch;
		for (int bi = 0; bi < Parameters::batch_size; bi++) {
			float* dev_bin;
			cudaMalloc((void**)&dev_bin, Parameters::num_points * Parameters::input_features * sizeof(float));
			dev_batch.push_back(dev_bin);
		}
		for (int bi = 0; bi < Parameters::batch_size; bi++) {
			float* dev_bin;
			cudaMalloc((void**)&dev_bin, Parameters::num_points * Parameters::num_points * sizeof(float));
			dev_batch.push_back(dev_bin);
		}
		memPrint("After Forward Input");

		
		// Iterate for as many epochs..
		for (int ep = 0; ep < Parameters::num_epochs; ep++) {
			std::cout << "****************************Epoch " << ep << "***************************" << std::endl;
			epochLoss = 0;
			memPrint("Memory Stats");

			// Loop batch by batch
			for (int b = 0; b < num_batches; b++) {
				
				// Grab one batch's data on CPU
				std::vector<float*> batch_in = std::vector < float* >(input.begin() + b * this->batchSize, input.begin() + (b + 1) * this->batchSize);
				std::vector<float*> trueLabel = std::vector < float* >(label.begin() + b * this->batchSize, label.begin() + (b + 1) * this->batchSize);

				for (int bi = 0; bi < batch_in.size(); bi++) {
					normalize_data(batch_in[bi], Parameters::num_points);
				}
				if (debug) {
					std::cout << "************************************************************************************" << std::endl;
					std::cout << "GPU : Forward for batch  ==> " << b << std::endl;
					std::cout << "************************************************************************************" << std::endl;
				}
				for (int bi = 0; bi < batch_in.size(); bi++) {
					cudaMemcpy(dev_in[bi], batch_in[bi], Parameters::num_points * Parameters::input_features * sizeof(float), cudaMemcpyHostToDevice);
				}
				
				auto start = high_resolution_clock::now();
				// Laplacian of the batch data
				get_laplacian(batch_in, dev_lap);

				auto stop = high_resolution_clock::now();
				auto duration = duration_cast<microseconds>(stop - start);
				if (time) {
					std::cout << "************************************************************************************" << std::endl;
					std::cout << "GPU : (Graph Generation / batch) ==> " << duration.count() / 1000 << " seconds" << std::endl;
					std::cout << "************************************************************************************" << std::endl;
				}

				// Copy Label and Forward Data
				for (int bi = 0; bi < batch_in.size(); bi++) {
					cudaMemcpy(dev_batch[bi], dev_in[bi], Parameters::num_points * Parameters::input_features * sizeof(float), cudaMemcpyDeviceToDevice);
				}
				for (int bi = batch_in.size(); bi < 2*batch_in.size(); bi++) {
					cudaMemcpy(dev_batch[bi], dev_lap[bi - batch_in.size()], Parameters::num_points * Parameters::num_points * sizeof(float), cudaMemcpyDeviceToDevice);
				}
				for (int bi = 0; bi < batch_in.size(); bi++) {
					cudaMemcpy(dev_label[bi], trueLabel[bi], Parameters::num_classes * sizeof(float), cudaMemcpyHostToDevice);
				}

				start = high_resolution_clock::now();
				// Forward Pass
				std::vector<float*> prediction = forward(dev_batch, false);
				stop = high_resolution_clock::now();
				duration = duration_cast<microseconds>(stop - start);
				if (time) {
					std::cout << "************************************************************************************" << std::endl;
					std::cout << "GPU : (Forward Pass / batch) ==> " << duration.count() / 1000 << " seconds" << std::endl;
					std::cout << "************************************************************************************" << std::endl;
				}

				// Calculate Loss
				float loss = calculateLoss(prediction, dev_label);
				epochLoss += loss;

				start = high_resolution_clock::now();
				PointCloudClassification::softmaxActivationLayerGPU softmaxLayer(numClasses, Parameters::batch_size, false);
				std::vector<float*> pprob = softmaxLayer.forward(prediction, false);

				// Check Prediction: Can comment this in training later on..
				getClassification(pprob, this->numClasses, classification);
				
				// Backward Pass
				if (debug) {
					std::cout << "************************************************************************************" << std::endl;
					std::cout << "GPU : Backward for batch  ==> " << b << std::endl;
					std::cout << "************************************************************************************" << std::endl;
				}
				backward(pprob, dev_label, Parameters::learning_rate);
				stop = high_resolution_clock::now();
				duration = duration_cast<microseconds>(stop - start);
				if (time) {
					std::cout << "************************************************************************************" << std::endl;
					std::cout << "GPU : (Backward Pass / batch) ==> " << duration.count() / 1000 << " seconds" << std::endl;
					std::cout << "************************************************************************************" << std::endl;
				}
			}
			epochLoss /= num_batches;
			perEpochLoss[ep] = epochLoss;
			std::cout << "Epoch: " << ep << " Loss: " << epochLoss << "\n";
		}
		std::cout << "Done with training, printing loss\n";
		Utilities::printArray(perEpochLoss, Parameters::num_epochs);
	}

	// Returns classification between [0, classes-1] for each instance
	void NetworkGPU::getClassification(const std::vector<float*> pprob, const int classes, std::vector<float> classification) {
		if (debug) {
			std::cout << "Actual Prediction: ";
			Utilities::printVectorOfFloatsGPU(pprob, Parameters::num_classes);
		}
		int n = pprob.size();
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
	}
	
}
