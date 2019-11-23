#pragma once
#include "../common.h"
#include "../network.h"
#include "../utilities/parameters.h"
#include "../hidden_layers/fullyConnectedLayerCPU.cu"
#include "../hidden_layers/RELUActivationLayerCPU.cu"
#include "../hidden_layers/softmaxActivationLayerCPU.cu"
#include "../hidden_layers/sigmoidActivationLayerCPU.cu"
#include "../hidden_layers/CrossEntropyLossCPU.cu"

using namespace std;
using namespace PointCloudClassification;
//using namespace Parameters;

namespace Tests {
	void testMatrixTranspose() {
		MatrixCPU* m = new MatrixCPU();
		float A[3 * 2] = { 1,2,3,4,5,6 };
		
		std::cout << "A: " << endl;
		m->printMatrix(A, 3, 2);
		std::cout << std::endl;

		float* B = (float*)malloc(3 * 2 * sizeof(float));
		m->transpose(A, 3, 2, B);

		std::cout << "A.T: " << endl;
		m->printMatrix(B, 2, 3);
		std::cout << std::endl;
		std::cout << std::endl;
	}
	
	void testMatrixMultiplication() {
		MatrixCPU* m = new MatrixCPU();
		float A[3 * 2] = { 1,2,3,4,5,6 };
		float B[2 * 5] = { 1,2,3,4,5,6, 7, 8, 9, 10 };
		float* C = (float*)malloc(3 * 5 * sizeof(float));
		m->multiply(A, B, 3, 2, 5, C);
		std::cout << "A: " << endl;
		m->printMatrix(A, 3, 2);
		std::cout << std::endl;

		std::cout << "B: " << endl;
		m->printMatrix(B, 2, 5);
		std::cout << std::endl;

		std::cout << "C = A X B : " << endl;
		m->printMatrix(C, 3, 5);
		std::cout << std::endl;
		std::cout << std::endl;
	}

	void testMatrixMultiplicationTranspose() {
		MatrixCPU* m = new MatrixCPU();
		float A[3 * 2] = { 1,2,3,4,5,6 };
		float B[5 * 2] = { 1,2,3,4,5,6, 7, 8, 9, 10 };
		float* C = (float*)malloc(3 * 5 * sizeof(float));
		m->multiplyTranspose(A, B, 3, 2, 5, C);
		std::cout << "A: " << endl;
		m->printMatrix(A, 3, 2);
		std::cout << std::endl;

		std::cout << "B: " << endl;
		m->printMatrix(B, 5, 2);
		std::cout << std::endl;

		std::cout << "C = A X B : " << endl;
		m->printMatrix(C, 3, 5);
		std::cout << std::endl;
		std::cout << std::endl;
	}


	void testFCLayer() {
		PointCloudClassification::NetworkCPU gcn(Parameters::num_points * Parameters::l1_features, Parameters::num_classes, Parameters::batch_size);
		PointCloudClassification::FullyConnectedLayerCPU fc1(Parameters::num_points * Parameters::l1_features, 1000, Parameters::batch_size, false);
		gcn.addLayer(&fc1);
		PointCloudClassification::FullyConnectedLayerCPU fc2(1000, 300, Parameters::batch_size, false);
		gcn.addLayer(&fc2);

		vector<float*> samples; //Data from file will be stored here
		int number_of_random_examples = Parameters::batch_size;
		for (int i = 0; i < number_of_random_examples; i++) {
			float* temp = (float*)malloc(Parameters::num_points * Parameters::l1_features * sizeof(float));
			Utilities::genArray(Parameters::num_points * Parameters::l1_features, temp);
			samples.push_back(temp);
		}

		std::cout << "SAMPLE: " << std::endl;
		std::cout << samples[0][0] << " " << samples[0][1] << " " << samples[0][2] << std::endl;
		std::cout << samples[1][0] << " " << samples[1][1] << " " << samples[1][2] << std::endl;
		std::cout << samples[2][0] << " " << samples[2][1] << " " << samples[2][2] << std::endl;
		std::cout << std::endl;

		std::vector<float*> op = gcn.forward(samples, false);

		std::cout << "OUTPUT: " << std::endl;
		std::cout << op[0][0] << " " << op[0][1] << " " << op[0][2] << std::endl;
		std::cout << op[1][0] << " " << op[1][1] << " " << op[1][2] << std::endl;
		std::cout << op[2][0] << " " << op[2][1] << " " << op[2][2] << std::endl;
		std::cout << std::endl;
	}

	void testRELULayer() {
		PointCloudClassification::NetworkCPU gcn(Parameters::num_points * Parameters::l1_features, Parameters::num_classes, Parameters::batch_size);

		PointCloudClassification::RELUActivationLayerCPU relu1(Parameters::num_points * Parameters::l1_features, Parameters::batch_size, false);
		gcn.addLayer(&relu1);

		vector<float*> samples; //Data from file will be stored here
		int number_of_random_examples = Parameters::batch_size;
		for (int i = 0; i < number_of_random_examples; i++) {
			float* temp = (float*)malloc(Parameters::num_points * Parameters::l1_features * sizeof(float));
			Utilities::genArray(Parameters::num_points * Parameters::l1_features, temp);
			samples.push_back(temp);
		}

		std::cout << "SAMPLE: " << std::endl;
		std::cout << samples[0][0] << " " << samples[0][1] << " " << samples[0][2] << std::endl;
		std::cout << samples[1][0] << " " << samples[1][1] << " " << samples[1][2] << std::endl;
		std::cout << samples[2][0] << " " << samples[2][1] << " " << samples[2][2] << std::endl;
		std::cout << std::endl;

		std::vector<float*> op = gcn.forward(samples, false);

		std::cout << "OUTPUT: " << std::endl;
		std::cout << op[0][0] << " " << op[0][1] << " " << op[0][2] << std::endl;
		std::cout << op[1][0] << " " << op[1][1] << " " << op[1][2] << std::endl;
		std::cout << op[2][0] << " " << op[2][1] << " " << op[2][2] << std::endl;
		std::cout << std::endl;
	}

	void testSigmoidLayer() {
		PointCloudClassification::NetworkCPU gcn(3, Parameters::num_classes, Parameters::batch_size);

		PointCloudClassification::sigmoidActivationLayerCPU sigmoid1(3, Parameters::batch_size, false);
		gcn.addLayer(&sigmoid1);

		vector<float*> samples; //Data from file will be stored here
		int number_of_random_examples = Parameters::batch_size;
		for (int i = 0; i < number_of_random_examples; i++) {
			float* temp = (float*)malloc(Parameters::num_points * Parameters::l1_features * sizeof(float));
			Utilities::genArray(Parameters::num_points * Parameters::l1_features, temp);
			samples.push_back(temp);
		}

		std::cout << "SAMPLE: " << std::endl;
		std::cout << samples[0][0] << " " << samples[0][1] << " " << samples[0][2] << std::endl;
		std::cout << samples[1][0] << " " << samples[1][1] << " " << samples[1][2] << std::endl;
		std::cout << samples[2][0] << " " << samples[2][1] << " " << samples[2][2] << std::endl;
		std::cout << std::endl;

		std::vector<float*> op = gcn.forward(samples, false);

		std::cout << "OUTPUT: " << std::endl;
		std::cout << op[0][0] << " " << op[0][1] << " " << op[0][2] << std::endl;
		std::cout << op[1][0] << " " << op[1][1] << " " << op[1][2] << std::endl;
		std::cout << op[2][0] << " " << op[2][1] << " " << op[2][2] << std::endl;
		std::cout << std::endl;
	}

	void testSoftmaxLayer() {
		PointCloudClassification::NetworkCPU gcn(3, Parameters::num_classes, Parameters::batch_size);

		PointCloudClassification::softmaxActivationLayerCPU softmax1 (3, Parameters::batch_size, false);
		gcn.addLayer(&softmax1);

		vector<float*> samples; //Data from file will be stored here
		int number_of_random_examples = Parameters::batch_size;
		for (int i = 0; i < number_of_random_examples; i++) {
			float* temp = (float*)malloc(Parameters::num_points * Parameters::l1_features * sizeof(float));
			Utilities::genArray(Parameters::num_points * Parameters::l1_features, temp);
			samples.push_back(temp);
		}

		std::cout << "SAMPLE: " << std::endl;
		std::cout << samples[0][0] << " " << samples[0][1] << " " << samples[0][2] << std::endl;
		std::cout << samples[1][0] << " " << samples[1][1] << " " << samples[1][2] << std::endl;
		std::cout << samples[2][0] << " " << samples[2][1] << " " << samples[2][2] << std::endl;
		std::cout << std::endl;

		std::vector<float*> op = gcn.forward(samples, false);

		std::cout << "OUTPUT: " << std::endl;
		std::cout << op[0][0] << " " << op[0][1] << " " << op[0][2] << std::endl;
		std::cout << op[1][0] << " " << op[1][1] << " " << op[1][2] << std::endl;
		std::cout << op[2][0] << " " << op[2][1] << " " << op[2][2] << std::endl;
		std::cout << std::endl;
	}

	void testCrossEntropyLoss() {
		PointCloudClassification::NetworkCPU gcn(3, Parameters::num_classes, Parameters::batch_size);

		PointCloudClassification::FullyConnectedLayerCPU fc1(3, 3, Parameters::batch_size, false);
		gcn.addLayer(&fc1);
		PointCloudClassification::CrossEntropyLossCPU celoss(Parameters::batch_size, 3);
		gcn.setLoss(&celoss);

		vector<float*> samples; //Data from file will be stored here
		vector<float*> trueLabels;
		float temp_true[3 * 3] = { 0, 1, 0, 1, 0, 0, 0, 0, 1 };
		int number_of_random_examples = Parameters::batch_size;
		for (int i = 0; i < number_of_random_examples; i++) {
			float* temp = (float*)malloc(3 * sizeof(float));
			
			Utilities::genArray(3, temp);
			samples.push_back(temp);

			trueLabels.push_back(temp_true + (i * 3));
		}

		

		std::cout << "SAMPLE: " << std::endl;
		std::cout << samples[0][0] << " " << samples[0][1] << " " << samples[0][2] << std::endl;
		std::cout << samples[1][0] << " " << samples[1][1] << " " << samples[1][2] << std::endl;
		std::cout << samples[2][0] << " " << samples[2][1] << " " << samples[2][2] << std::endl;
		std::cout << std::endl;

		std::cout << "TRUE LABELS: " << std::endl;
		std::cout << trueLabels[0][0] << " " << trueLabels[0][1] << " " << trueLabels[0][2] << std::endl;
		std::cout << trueLabels[1][0] << " " << trueLabels[1][1] << " " << trueLabels[1][2] << std::endl;
		std::cout << trueLabels[2][0] << " " << trueLabels[2][1] << " " << trueLabels[2][2] << std::endl;
		std::cout << std::endl;

		std::vector<float*> op = gcn.forward(samples, false);

		std::cout << "OUTPUT: " << std::endl;
		std::cout << op[0][0] << " " << op[0][1] << " " << op[0][2] << std::endl;
		std::cout << op[1][0] << " " << op[1][1] << " " << op[1][2] << std::endl;
		std::cout << op[2][0] << " " << op[2][1] << " " << op[2][2] << std::endl;
		std::cout << std::endl;

		float l = gcn.calculateLoss(op, trueLabels);
		std::cout << "LOSS: " << l << std::endl;
	}

	void testAllBackward() {
		PointCloudClassification::NetworkCPU gcn(Parameters::num_points * Parameters::l1_features, Parameters::num_classes, Parameters::batch_size);
		PointCloudClassification::FullyConnectedLayerCPU fc1(Parameters::num_points * Parameters::l1_features, 100, Parameters::batch_size, false);
		gcn.addLayer(&fc1);
		PointCloudClassification::RELUActivationLayerCPU relu1(100, Parameters::batch_size, false);
		gcn.addLayer(&relu1);
		PointCloudClassification::FullyConnectedLayerCPU fc2(100, 3, Parameters::batch_size, false);
		gcn.addLayer(&fc2);
		
		PointCloudClassification::CrossEntropyLossCPU celoss(Parameters::batch_size, 3);
		gcn.setLoss(&celoss);

		vector<float*> samples; //Data from file will be stored here
		vector<float*> trueLabels;
		float temp_true[3 * 3] = { 0, 1, 0, 1, 0, 0, 0, 0, 1 };
		int number_of_random_examples = Parameters::batch_size;
		for (int i = 0; i < number_of_random_examples; i++) {
			float* temp = (float*)malloc(Parameters::num_points * Parameters::l1_features * sizeof(float));

			Utilities::genArray(Parameters::num_points * Parameters::l1_features, temp);
			samples.push_back(temp);

			trueLabels.push_back(temp_true + (i * 3));
		}

		std::cout << "SAMPLE: " << std::endl;
		std::cout << samples[0][0] << " " << samples[0][1] << " " << samples[0][2] << std::endl;
		std::cout << samples[1][0] << " " << samples[1][1] << " " << samples[1][2] << std::endl;
		std::cout << samples[2][0] << " " << samples[2][1] << " " << samples[2][2] << std::endl;
		std::cout << std::endl;

		std::vector<float*> op = gcn.forward(samples, false);

		float l = gcn.calculateLoss(op, trueLabels);
		std::cout << "*****LOSS: " << l << std::endl;

		gcn.backward(op, trueLabels, 0.01);

		op = gcn.forward(samples, false);

		std::cout << "NEW OUTPUT: " << std::endl;
		std::cout << op[0][0] << " " << op[0][1] << " " << op[0][2] << std::endl;
		std::cout << op[1][0] << " " << op[1][1] << " " << op[1][2] << std::endl;
		std::cout << op[2][0] << " " << op[2][1] << " " << op[2][2] << std::endl;
		std::cout << std::endl;

		l = gcn.calculateLoss(op, trueLabels);
		std::cout << "******NEW LOSS: " << l << std::endl;

		gcn.backward(op, trueLabels, 0.01);

		op = gcn.forward(samples, false);

		std::cout << "NEW OUTPUT: " << std::endl;
		std::cout << op[0][0] << " " << op[0][1] << " " << op[0][2] << std::endl;
		std::cout << op[1][0] << " " << op[1][1] << " " << op[1][2] << std::endl;
		std::cout << op[2][0] << " " << op[2][1] << " " << op[2][2] << std::endl;
		std::cout << std::endl;

		l = gcn.calculateLoss(op, trueLabels);
		std::cout << "*******NEW LOSS: " << l << std::endl;

		gcn.backward(op, trueLabels, 0.01);

		op = gcn.forward(samples, false);

		std::cout << "NEW OUTPUT: " << std::endl;
		std::cout << op[0][0] << " " << op[0][1] << " " << op[0][2] << std::endl;
		std::cout << op[1][0] << " " << op[1][1] << " " << op[1][2] << std::endl;
		std::cout << op[2][0] << " " << op[2][1] << " " << op[2][2] << std::endl;
		std::cout << std::endl;

		l = gcn.calculateLoss(op, trueLabels);
		std::cout << "*****NEW LOSS: " << l << std::endl;
	}
}