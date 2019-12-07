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

int main(int argc, char* argv[]) {

	//// Tests
	//tests();

	// Read data from file and store it as a vector of float pointers (length of vector -> number of samples | each sample -> 1024 x 3 floats)
	int per_class = 32*4;
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


