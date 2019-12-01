#pragma once
#include "../common.h"
#include "../network.h"
#include "../utilities/parameters.h"
#include "../hidden_layers/fullyConnectedLayerCPU.cu"
#include "../hidden_layers/fullyConnectedLayerGPU.cu"
#include "../hidden_layers/globalPoolingCPU.cu"
#include "../hidden_layers/graphConvolutionLayerCPU.cu"
#include "../hidden_layers/RELUActivationLayerCPU.cu"
#include "../hidden_layers/softmaxActivationLayerCPU.cu"
#include "../hidden_layers/sigmoidActivationLayerCPU.cu"
#include "../hidden_layers/CrossEntropyLossCPU.cu"
#include "../hidden_layers/dropoutLayerCPU.cu"

using namespace std;
using namespace PointCloudClassification;

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
		PointCloudClassification::FullyConnectedLayerCPU fc1(Parameters::num_points * Parameters::input_features, 1000, Parameters::batch_size, false);

		vector<float*> samples; //Data from file will be stored here
		int number_of_random_examples = Parameters::batch_size;
		for (int i = 0; i < number_of_random_examples; i++) {
			float* temp = (float*)malloc(Parameters::num_points * Parameters::input_features * sizeof(float));
			Utilities::genArray(Parameters::num_points * Parameters::input_features, temp);
			samples.push_back(temp);
		}

		std::cout << "SAMPLE: " << std::endl;
		Utilities::printVectorOfFloats(samples, 5);
		std::cout << std::endl;

		std::vector<float*> op = fc1.forward(samples, false);

		std::cout << "OUTPUT: " << std::endl;
		Utilities::printVectorOfFloats(op, 5);
		std::cout << std::endl;
	}

	void testFCLayerGPU() {
		PointCloudClassification::FullyConnectedLayerGPU fc1(Parameters::num_points * Parameters::input_features, 1000, Parameters::batch_size, false);

		vector<float*> samples; //Data from file will be stored here
		int number_of_random_examples = Parameters::batch_size;
		for (int i = 0; i < number_of_random_examples; i++) {
			float* temp = (float*)malloc(Parameters::num_points * Parameters::input_features * sizeof(float));
			Utilities::genArray(Parameters::num_points * Parameters::input_features, temp);
			float* temp_gpu;
			cudaMalloc((void**)&temp_gpu, Parameters::num_points * Parameters::input_features * sizeof(float));
			cudaMemcpy(temp_gpu, temp, Parameters::num_points * Parameters::input_features * sizeof(float), cudaMemcpyHostToDevice);
			samples.push_back(temp_gpu);
		}

		//std::cout << "SAMPLE: " << std::endl;
		//std::cout << samples[0][0] << " " << samples[0][1] << " " << samples[0][2] << std::endl;
		//std::cout << samples[1][0] << " " << samples[1][1] << " " << samples[1][2] << std::endl;
		//std::cout << samples[2][0] << " " << samples[2][1] << " " << samples[2][2] << std::endl;
		//std::cout << std::endl;

		std::cout << "INPUT: " << std::endl;
		Utilities::printVectorOfFloatsGPU(samples, 5);
		std::cout << std::endl;

		std::vector<float*> op = fc1.forward(samples, false);

		std::cout << "OUTPUT: " << std::endl;
		Utilities::printVectorOfFloatsGPU(op, 5);
		std::cout << std::endl;
	}

	void testFCLayerBackwardGPU() {
		PointCloudClassification::FullyConnectedLayerGPU fc1(Parameters::num_points * Parameters::input_features, 1000, Parameters::batch_size, false);

		vector<float*> samples; //Data from file will be stored here
		int number_of_random_examples = Parameters::batch_size;
		for (int i = 0; i < number_of_random_examples; i++) {
			float* temp = (float*)malloc(Parameters::num_points * Parameters::input_features * sizeof(float));
			Utilities::genArray(Parameters::num_points * Parameters::input_features, temp);
			float* temp_gpu;
			cudaMalloc((void**)&temp_gpu, Parameters::num_points * Parameters::input_features * sizeof(float));
			cudaMemcpy(temp_gpu, temp, Parameters::num_points * Parameters::input_features * sizeof(float), cudaMemcpyHostToDevice);
			samples.push_back(temp_gpu);
		}

		//std::cout << "SAMPLE: " << std::endl;
		//std::cout << samples[0][0] << " " << samples[0][1] << " " << samples[0][2] << std::endl;
		//std::cout << samples[1][0] << " " << samples[1][1] << " " << samples[1][2] << std::endl;
		//std::cout << samples[2][0] << " " << samples[2][1] << " " << samples[2][2] << std::endl;
		//std::cout << std::endl;

		std::cout << "INPUT: " << std::endl;
		Utilities::printVectorOfFloatsGPU(samples, 5);
		std::cout << std::endl;

		std::vector<float*> op = fc1.forward(samples, false);

		std::cout << "OUTPUT: " << std::endl;
		Utilities::printVectorOfFloatsGPU(op, 5);
		std::cout << std::endl;

		std::vector<float*> og = fc1.backward(op, 0.01);

		

		std::cout << "OG: " << std::endl;
		Utilities::printVectorOfFloatsGPU(og, 5);
		std::cout << std::endl;
	}

	void testGlobalPoolingLayer() {
		int pts = 5;
		PointCloudClassification::GlobalPoolingLayerCPU gp_layer(pts, Parameters::input_features, Parameters::batch_size, false);

		vector<float*> samples; //Data from file will be stored here
		int number_of_random_examples = Parameters::batch_size;
		for (int i = 0; i < number_of_random_examples; i++) {
			float temp[15] = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,11.0f,12.0f,13.0f,14.0f,15.0f };
			samples.push_back(temp);
		}

		std::cout << "SAMPLE: " << std::endl;
		std::cout << samples[0][0] << " " << samples[0][1] << " " << samples[0][2] << std::endl;
		std::cout << samples[0][3] << " " << samples[0][4] << " " << samples[0][5] << std::endl;
		std::cout << samples[0][6] << " " << samples[0][7] << " " << samples[0][8] << std::endl;
		std::cout << samples[0][9] << " " << samples[0][10] << " " << samples[0][11] << std::endl;
		std::cout << samples[0][12] << " " << samples[0][13] << " " << samples[0][14] << std::endl;
		std::cout << std::endl;

		std::vector<float*> op = gp_layer.forward(samples, false);

		std::cout << "MAX POOL OUTPUT: " << std::endl; // Should be 13 14 15
		std::cout << op[0][0] << " " << op[0][1] << " " << op[0][2] << std::endl;
		std::cout << "VARIANCE POOL OUTPUT: " << std::endl; // Should be 18 18 18
		std::cout << op[0][3] << " " << op[0][4] << " " << op[0][5] << std::endl;
		std::cout << std::endl;

		std::vector<float*> outGradient = gp_layer.backward(op, 1);
		
		std::cout << "GRADIENT: " << std::endl;
		std::cout << outGradient[0][0] << " " << outGradient[0][1] << " " << outGradient[0][2] << std::endl; // -43.2 -43.2 -43.2
		std::cout << outGradient[0][3] << " " << outGradient[0][4] << " " << outGradient[0][5] << std::endl; //	-21.6 -21.6 -21.6
		std::cout << outGradient[0][6] << " " << outGradient[0][7] << " " << outGradient[0][8] << std::endl; //	0 0 0
		std::cout << outGradient[0][9] << " " << outGradient[0][10] << " " << outGradient[0][11] << std::endl; // 21.6 21.6 21.6
		std::cout << outGradient[0][12] << " " << outGradient[0][13] << " " << outGradient[0][14] << std::endl; // 56.2 57.2 58.2
		std::cout << std::endl;
	}

	void testDropoutLayer() {
		int pts = 5;
		PointCloudClassification::DropoutLayerCPU dropout_layer(pts, Parameters::input_features, Parameters::batch_size, false, Parameters::keep_drop_prob1);

		vector<float*> samples; //Data from file will be stored here
		int number_of_random_examples = Parameters::batch_size;
		for (int i = 0; i < number_of_random_examples; i++) {
			float temp[15] = { 1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,11.0f,12.0f,13.0f,14.0f,15.0f };
			samples.push_back(temp);
		}

		std::cout << "SAMPLE: " << std::endl;
		std::cout << samples[0][0] << " " << samples[0][1] << " " << samples[0][2] << std::endl;
		std::cout << samples[0][3] << " " << samples[0][4] << " " << samples[0][5] << std::endl;
		std::cout << samples[0][6] << " " << samples[0][7] << " " << samples[0][8] << std::endl;
		std::cout << samples[0][9] << " " << samples[0][10] << " " << samples[0][11] << std::endl;
		std::cout << samples[0][12] << " " << samples[0][13] << " " << samples[0][14] << std::endl;
		std::cout << std::endl;

		std::vector<float*> op = dropout_layer.forward(samples, false);

		std::cout << "OUTPUT: " << std::endl;
		std::cout << op[0][0] << " " << op[0][1] << " " << op[0][2] << std::endl;
		std::cout << op[0][3] << " " << op[0][4] << " " << op[0][5] << std::endl;
		std::cout << op[0][6] << " " << op[0][7] << " " << op[0][8] << std::endl;
		std::cout << op[0][9] << " " << op[0][10] << " " << op[0][11] << std::endl;
		std::cout << op[0][12] << " " << op[0][13] << " " << op[0][14] << std::endl;
		std::cout << std::endl;

		std::vector<float*> outGradient = dropout_layer.backward(op, 1);

		std::cout << "GRADIENT: " << std::endl;
		std::cout << outGradient[0][0] << " " << outGradient[0][1] << " " << outGradient[0][2] << std::endl; // -43.2 -43.2 -43.2
		std::cout << outGradient[0][3] << " " << outGradient[0][4] << " " << outGradient[0][5] << std::endl; //	-21.6 -21.6 -21.6
		std::cout << outGradient[0][6] << " " << outGradient[0][7] << " " << outGradient[0][8] << std::endl; //	0 0 0
		std::cout << outGradient[0][9] << " " << outGradient[0][10] << " " << outGradient[0][11] << std::endl; // 21.6 21.6 21.6
		std::cout << outGradient[0][12] << " " << outGradient[0][13] << " " << outGradient[0][14] << std::endl; // 56.2 57.2 58.2
		std::cout << std::endl;
	}
	void testRELULayer() {
		PointCloudClassification::NetworkCPU gcn(Parameters::num_classes, Parameters::batch_size);

		PointCloudClassification::RELUActivationLayerCPU relu1(Parameters::num_points * Parameters::input_features, Parameters::batch_size, false);
		gcn.addLayer(&relu1);

		vector<float*> samples; //Data from file will be stored here
		int number_of_random_examples = Parameters::batch_size;
		for (int i = 0; i < number_of_random_examples; i++) {
			float* temp = (float*)malloc(Parameters::num_points * Parameters::input_features * sizeof(float));
			Utilities::genArray(Parameters::num_points * Parameters::input_features, temp);
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
		PointCloudClassification::NetworkCPU gcn(Parameters::num_classes, Parameters::batch_size);

		PointCloudClassification::sigmoidActivationLayerCPU sigmoid1(3, Parameters::batch_size, false);
		gcn.addLayer(&sigmoid1);

		vector<float*> samples; //Data from file will be stored here
		int number_of_random_examples = Parameters::batch_size;
		for (int i = 0; i < number_of_random_examples; i++) {
			float* temp = (float*)malloc(Parameters::num_points * Parameters::input_features * sizeof(float));
			Utilities::genArray(Parameters::num_points * Parameters::input_features, temp);
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
		PointCloudClassification::NetworkCPU gcn(Parameters::num_classes, Parameters::batch_size);

		PointCloudClassification::softmaxActivationLayerCPU softmax1 (3, Parameters::batch_size, false);
		gcn.addLayer(&softmax1);

		vector<float*> samples; //Data from file will be stored here
		int number_of_random_examples = Parameters::batch_size;
		for (int i = 0; i < number_of_random_examples; i++) {
			float* temp = (float*)malloc(Parameters::num_points * Parameters::input_features * sizeof(float));
			Utilities::genArray(Parameters::num_points * Parameters::input_features, temp);
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
		PointCloudClassification::NetworkCPU gcn(Parameters::num_classes, Parameters::batch_size);

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
		PointCloudClassification::NetworkCPU gcn(Parameters::num_classes, Parameters::batch_size);
		PointCloudClassification::FullyConnectedLayerCPU fc1(Parameters::num_points * Parameters::input_features, 100, Parameters::batch_size, false);
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
			float* temp = (float*)malloc(Parameters::num_points * Parameters::input_features * sizeof(float));

			Utilities::genArray(Parameters::num_points * Parameters::input_features, temp);
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