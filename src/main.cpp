/**
 * @file      main.cpp
 * @brief     Point Cloud Classification
 * @authors   Kushagra Goel, Saket Karve, Disha Jindal
 * @date      2019
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <point_cloud_classification/network.h>
#include <point_cloud_classification/common.h>
#include "utilityCore.hpp"
#include "point_cloud_classification/graph/graph.h"
#include "point_cloud_classification/utilities/matrix.h"
#include "point_cloud_classification/utilities/utils.h"
#include "point_cloud_classification/utilities/parameters.h"
#include "point_cloud_classification/hidden_layers/fullyConnectedLayerCPU.cu"

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

void tests() {
	cout << "********************************************************" << endl;
	cout << "Testing Matrix Transpose ..." << endl;
	Tests::testMatrixTranspose();
	cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing Matrix Multiplication ..." << endl;
	Tests::testMatrixMultiplication();
	cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing Matrix Multiplication Transpose ..." << endl;
	Tests::testMatrixMultiplicationTranspose();
	cout << "********************************************************" << endl;

	//cout << "********************************************************" << endl;
	//cout << "Testing Fully Connected Layer ..." << endl;
	//Tests::testFCLayer();
	//cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing RELU Activation Layer ..." << endl;
	Tests::testRELULayer();
	cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing Softmax Activation Layer ..." << endl;
	Tests::testSoftmaxLayer();
	cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing Sigmoid Activation Layer ..." << endl;
	Tests::testSigmoidLayer();
	cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing Cross Entropy Layer ..." << endl;
	Tests::testCrossEntropyLoss();
	cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing All Backward..." << endl;
	Tests::testAllBackward();
	cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing Global Pooling Layer ..." << endl;
	Tests::testGlobalPoolingLayer();
	cout << "********************************************************" << endl;

	cout << "********************************************************" << endl;
	cout << "Testing Dropout Layer ..." << endl;
	Tests::testDropoutLayer();
	cout << "********************************************************" << endl;
}

int main(int argc, char* argv[]) {

	// Tests
	//tests();

	// Read data from file and store it as a vector of float pointers (length of vector -> number of samples | each sample -> 1024 x 3 floats)

	int per_class = 1;
	std::vector<float*> x_train;
	std::vector<float*> y_train;
	utilityCore::load_data("bullshit", x_train, y_train, "train", per_class);
	std::cout << "Loaded Data: " << x_train.size() << std::endl;
	// Construct graph for each example and store a vector of L (Laplacians) and AX for each sample
	vector<float*> laplacians;

	for (int i = 0; i < x_train.size(); i++) {
		float* current_sample = x_train[i];
		utilityCore::normalize_data(current_sample, Parameters::num_points);
		Graph::Graph g (current_sample, Parameters::num_points, Parameters::input_features, Parameters::num_neighbours);
		std::cout << "Constructed graph for " << i << std::endl;
		//float* A = g.get_A();
		//MatrixCPU* m = new MatrixCPU();
		//float* AX = (float*)malloc(Parameters::num_points * Parameters::input_features * sizeof(float));
		//m->multiply(A, current_sample, Parameters::num_points, Parameters::num_points, Parameters::input_features, AX);

		float* L = g.get_Lnorm();
		laplacians.push_back(L);
	}

	
	//Build the network
	PointCloudClassification::NetworkCPU gcn(Parameters::num_classes, Parameters::batch_size);
	gcn.buildArchitecture();
	std::cout << "Built Architecture!" << std::endl;

	//:train(std::vector<float*> input, std::vector<float*> label, int n)
	gcn.train(x_train, laplacians, y_train, x_train.size());

	// Train 
	//int number_of_batches = ceil(Parameters::num_points / Parameters::batch_size);
}


