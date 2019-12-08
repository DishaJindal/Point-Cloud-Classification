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
#include "visualization.hpp"
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
#define TEST true

int main(int argc, char* argv[]) {

	//// Tests
	//tests();

	// Read data from file and store it as a vector of float pointers (length of vector -> number of samples | each sample -> 1024 x 3 floats)
	int per_class = 10;
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
	int count_loaded = utilityCore::load_data("bullshit", x_train, y_train, "train", per_class);
	x_train.erase(x_train.begin() + count_loaded, x_train.end());
	y_train.erase(y_train.begin() + count_loaded, y_train.end());
	std::cout << "Loaded Data: " << x_train.size() << std::endl;

	int index = 9;
	visualize(x_train[index], 1024, distance(y_train[index], max_element(y_train[index], y_train[index] + 10)), -1);


	//Build the network
	if (GPU) {
		PointCloudClassification::NetworkGPU gcn(Parameters::num_classes, Parameters::batch_size);
		gcn.buildArchitecture();
		PointCloudClassification::CrossEntropyLossGPU celoss(Parameters::batch_size, Parameters::num_classes);
		gcn.setLoss(&celoss);
		std::cout << "Built Architecture!" << std::endl;
		gcn.train(x_train, y_train, x_train.size());
		if (TEST) {
			gcn.dropout_layer1.batchDim = 1;
			gcn.dropout_layer2.batchDim = 1;
			gcn.dropout_layer3.batchDim = 1;
			gcn.dropout_layer4.batchDim = 1;
			gcn.fc_layer1.batchDim = 1;
			gcn.fc_layer2.batchDim = 1;
			gcn.gcn_layer1.batchDim = 1;
			gcn.gcn_layer2.batchDim = 1;
			gcn.gp_layer1.batchDim = 1;
			gcn.gp_layer2.batchDim = 1;
			gcn.relu1.batchDim = 1;
			
			std::vector<float> classification;
			int number_of_test_samples = 8;
			for (int i = 0; i < number_of_test_samples; i++) {
				//visualize(x_train[i], 1024, distance(y_train[i], max_element(y_train[i], y_train[i] + 10)), -1);
				std::vector<float*> batch_in;
				batch_in.push_back(x_train[i]);
				std::vector<float*> trueLabel;
				trueLabel.push_back(y_train[i]);
				classification = gcn.test(batch_in, trueLabel);
				//visualize(x_train[i], 1024, distance(y_train[index], max_element(y_train[index], y_train[index] + 10)), classification[i]);
			}
			
		}
	}
	else {
		PointCloudClassification::NetworkCPU gcn(Parameters::num_classes, Parameters::batch_size);
		gcn.buildArchitecture();
		PointCloudClassification::CrossEntropyLossCPU celoss(Parameters::batch_size, Parameters::num_classes);
		gcn.setLoss(&celoss);
		std::cout << "Built Architecture!" << std::endl;
		gcn.train(x_train, y_train, x_train.size());
	}

	visualize(x_train[0], 1024, distance(y_train[0], max_element(y_train[0], y_train[0] + 10)), -1);
}


