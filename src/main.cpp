/**
 * @file      main.cpp
 * @brief     Point Cloud Classification
 * @authors   Kushagra Goel, Saket Karve, Disha Jindal
 * @date      2019
 * @copyright University of Pennsylvania
 */
#pragma once

#include <cstdio>
#include <point_cloud_classification/network.h>
#include <point_cloud_classification/common.h>
#include "utilityCore.hpp"
#include "point_cloud_classification/graph/graph.h"
#include "point_cloud_classification/utilities/matrix.h"
#include "point_cloud_classification/utilities/utils.h"
#include "point_cloud_classification/utilities/parameters.h"
#include "point_cloud_classification/hidden_layers/fullyConnectedLayer.h"
#include "point_cloud_classification/hidden_layers/crossEntropyLoss.h"
#include "point_cloud_classification/tests/test.h"

#include <fstream>
#include <string>
#include <Windows.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <math.h>
using namespace std;
using namespace PointCloudClassification;
#define GPU true

void tests() {
	//cout << "********************************************************" << endl;
	//cout << "Testing GPU Matrix Reduction ..." << endl;
	//Tests::testMatrixGPUReduction();
	//cout << "********************************************************" << endl;

	//cout << "********************************************************" << endl;
	//cout << "Testing Matrix Transpose ..." << endl;
	//Tests::testMatrixTranspose();
	//cout << "********************************************************" << endl;

	//cout << "********************************************************" << endl;
	//cout << "Testing Matrix Multiplication ..." << endl;
	//Tests::testMatrixMultiplication();
	//cout << "********************************************************" << endl;

	//cout << "********************************************************" << endl;
	//cout << "Testing Matrix Multiplication Transpose ..." << endl;
	//Tests::testMatrixMultiplicationTranspose();
	//cout << "********************************************************" << endl;

	////cout << "********************************************************" << endl;
	////cout << "Testing Fully Connected Layer ..." << endl;
	////Tests::testFCLayer();
	////cout << "********************************************************" << endl;

	//cout << "********************************************************" << endl;
	//cout << "Testing RELU Activation Layer ..." << endl;
	//Tests::testRELULayer();
	//cout << "********************************************************" << endl;

	//cout << "********************************************************" << endl;
	//cout << "Testing Softmax Activation Layer ..." << endl;
	//Tests::testSoftmaxLayer();
	//cout << "********************************************************" << endl;

	//cout << "********************************************************" << endl;
	//cout << "Testing Sigmoid Activation Layer ..." << endl;
	//Tests::testSigmoidLayer();
	//cout << "********************************************************" << endl;

	//cout << "********************************************************" << endl;
	//cout << "Testing Cross Entropy Layer ..." << endl;
	//Tests::testCrossEntropyLoss();
	//cout << "********************************************************" << endl;

	//cout << "********************************************************" << endl;
	//cout << "Testing All Backward..." << endl;
	//Tests::testAllBackward();
	//cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing Global Pooling Layer ..." << endl;
	Tests::testGlobalPoolingLayer();
	cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing Softmax Layer forward GPU..." << endl;
	Tests::testSoftmaxLayerGPU();
	cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing RELU Layer forward CPU..." << endl;
	Tests::testRELULayer();
	cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing RELU Layer forward GPU..." << endl;
	Tests::testRELULayerGPU();
	cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing CE LOSS GPU..." << endl;
	Tests::testCrossEntropyLossGPU();
	cout << "********************************************************" << endl; 

	cout << "********************************************************" << endl;
	cout << "Testing Dropout Layer ..." << endl;
	Tests::testDropoutLayer();
	cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing Droput Layer GPU..." << endl;
	Tests::testDropoutLayerGPU();
	cout << "********************************************************" << endl;
	/*cout << "********************************************************" << endl;
	cout << "Testing Fully Connected Layer CPU ..." << endl;
	Tests::testFCLayer();
	cout << "********************************************************" << endl;

	//cout << "********************************************************" << endl;
	//cout << "Testing Fully Connected Layer GPU..." << endl;
	//Tests::testFCLayerGPU();
	//cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing Fully Connected Layer backward GPU..." << endl;
	Tests::testFCLayerBackwardGPU();
	cout << "********************************************************" << endl;

	//cout << "********************************************************" << endl;
	//cout << "Testing GPU Matrix Reduction ..." << endl;
	//Tests::testMatrixGPUReduction();
	//cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing Graph Convolutional Layer backward CPU..." << endl;
	Tests::testGraphConvolutionLayer();
	cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing Graph Convolutional Layer forward GPU..." << endl;
	Tests::testGraphConvolutionLayerGPU();
	cout << "********************************************************" << endl;


	cout << "********************************************************" << endl;
	cout << "Testing Softmax Layer forward CPU..." << endl;
	Tests::testSoftmaxLayer();
	cout << "********************************************************" << endl;*/
}

int main(int argc, char* argv[]) {

	// Tests
	//tests();

	/*cout << "********************************************************" << endl;
	cout << "Testing Graph Convolutional Layer forward GPU..." << endl;
	Tests::testGraphConvolutionLayerGPU();
	cout << "********************************************************" << endl;*/

	// Read data from file and store it as a vector of float pointers (length of vector -> number of samples | each sample -> 1024 x 3 floats)
	int per_class = 5;
	std::vector<float*> x_train;
	std::vector<float*> y_train;

	for (int i = 0; i < per_class * Parameters::num_classes; i++) {
		float* x_temp = (float*)malloc(Parameters::num_points * Parameters::input_features * sizeof(float));
		x_train.push_back(x_temp);
		float* y_temp = (float*)malloc(Parameters::num_classes * sizeof(float));
		memset(y_temp, 0.0f, Parameters::num_classes * sizeof(float));
		y_train.push_back(y_temp);
	}
	// Make sure you have downloaded the data
	utilityCore::load_data("bullshit", x_train, y_train, "train", per_class);
	std::cout << "Loaded Data: " << x_train.size() << std::endl;
	
	//Build the network
	if (GPU) {
		PointCloudClassification::NetworkGPU gcn(Parameters::num_classes, Parameters::batch_size);
		gcn.buildArchitecture();
		PointCloudClassification::CrossEntropyLossGPU celoss(Parameters::batch_size, Parameters::num_classes);
		gcn.setLoss(&celoss);
		std::cout << "Built Architecture!" << std::endl;
		gcn.train(x_train, y_train, per_class * 10);
	}
	else {
		PointCloudClassification::NetworkCPU gcn(Parameters::num_classes, Parameters::batch_size);
		gcn.buildArchitecture();
		PointCloudClassification::CrossEntropyLossCPU celoss(Parameters::batch_size, Parameters::num_classes);
		gcn.setLoss(&celoss);
		std::cout << "Built Architecture!" << std::endl;
		gcn.train(x_train, y_train, per_class * 10);
	}
}


