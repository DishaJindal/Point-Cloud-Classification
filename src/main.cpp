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
#include "testing_helpers.hpp"
#include <fstream>
#include <string>
#include <Windows.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <math.h>

using namespace::std;

// read MNIST data into double vector, OpenCV Mat, or Armadillo mat
// free to use this code for any purpose
// author : Eric Yuan 
// my blog: http://eric-yuan.me/
// part of this code is stolen from http://compvisionlab.wordpress.com/

#include <math.h>
#include <iostream>

using namespace std;


int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(string filename, vector<vector<double> > &vec)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < number_of_images; ++i)
		{
			vector<double> tp;
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					tp.push_back((double)temp);
				}
			}
			vec.push_back(tp);
		}
	}
	else {
		cout << "Couldn't Open"<<endl;
	}
}

void read_Mnist_Label(string filename, vector<double> &vec)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		for (int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			vec[i] = (double)temp;
		}
	}
}

int main(int argc, char* argv[]) {

	bool test = true;
	int seed = 36;


	ofstream mnistData;
	mnistData.open(R"(..\bookKeeping\mnistLosses.csv)");
	string filename = R"(..\data-set\mnist\t10k-images-idx3-ubyte)";
	int number_of_images = 10000;
	int number_of_images_considered = 100;
	int image_size = 28 * 28;
	int number_of_classes = 10;
	int numEpochs = 100;
	int hiddenDimensionsMnist[] = { 64 , 32 };
	PointCloudClassification::GraphConvolutionNetworkGPU *mnistMlp = new PointCloudClassification::GraphConvolutionNetworkGPU(image_size + 1, 2, hiddenDimensionsMnist, number_of_classes, number_of_images_considered);

	float *inputMnist;
	inputMnist = new float[number_of_images * (image_size + 1)];

	float *labelMnist;
	labelMnist = new float[number_of_images*number_of_classes];

	float *predictedMnist;
	predictedMnist = new float[number_of_images*number_of_classes];

	//read MNIST image into double vector
	vector<vector<double> > vec;
	read_Mnist(filename, vec);

	for (int i = 0; i < number_of_images; i++) {
		for (int j = 0; j < image_size + 1; j++) {
			if (j == image_size) {
				inputMnist[i * image_size + j] = 1;
				continue;
			}
			inputMnist[i * image_size + j] = (vec[i][j] - 0.5) * 2;
		}
	}

	filename = R"(..\data-set\mnist\t10k-labels-idx1-ubyte)";
	//read MNIST label into double vector
	vector<double> vecLabel(number_of_images);
	read_Mnist_Label(filename, vecLabel);


	for (int i = 0; i < number_of_images; i++) {
		for (int j = 0; j < number_of_classes; j++) {
			labelMnist[i * number_of_classes + j] = (j == vecLabel[i]) ? 1.0 : 0.0;
		}
	}

	printf("\n");
	printf("****************\n");
	printf("**** MNIST *****\n");
	printf("****************\n");
	cout << "\tMNIST Training : " << endl;
	for (int i = 0; i < numEpochs; i++) {
		mnistMlp->forward(inputMnist, predictedMnist);
		float loss = mnistMlp->loss(labelMnist, predictedMnist);
		if (!isnan(loss)) {
			mnistData << loss << endl;
		}
		mnistMlp->backward(labelMnist, predictedMnist, 0.0001);
	}
	cout << "\t\tFinal Loss : " << mnistMlp->loss(labelMnist, predictedMnist) << endl;

	cout << "\n\tMNIST Testing : " << endl;
	mnistMlp->forward(inputMnist, predictedMnist, false);
	srand(seed);
	for (int k = 0; k < 5; k++) {
		float maxTrueValue = 0;
		int maxTrueIndex = 0;
		float maxValue = 0;
		int maxIndex = 0;
		int i = rand() % number_of_images_considered;
		for (int j = 0; j < number_of_classes; j++) {

			if (predictedMnist[i * number_of_classes + j] > maxValue) {
				maxValue = predictedMnist[i * number_of_classes + j];
				maxIndex = j;
			}
			if (labelMnist[i * number_of_classes + j] > maxTrueValue) {
				maxTrueValue = predictedMnist[i * number_of_classes + j];
				maxTrueIndex = j;
			}
		}
		cout << "\t\tFor True Label = " << maxTrueIndex;
		cout << ", Predicted = " << maxIndex << endl;
	}


	numEpochs = 1000;
	int hiddenDimensions[] = { 5 };
	PointCloudClassification::GraphConvolutionNetworkGPU *mlp = new PointCloudClassification::GraphConvolutionNetworkGPU(3, 1, hiddenDimensions, 2, 4);
	ofstream xorData;
	xorData.open(R"(..\bookKeeping\xorLosses.csv)");

	float inputs[] = { 0, 0, 1, 
						0, 1, 1, 
						1, 0, 1, 
						1, 1, 1};

	float labels[] = { 1, 0,
						0, 1,
						0, 1,
						1, 0 };

	float *predicted = new float[8];


	printf("\n");
	printf("****************\n");
	printf("***** XOR ******\n");
	printf("****************\n");
	cout << "\tXOR Training : " << endl;
	for (int i = 0; i < numEpochs; i++) {
		mlp->forward(inputs, predicted);
		xorData << mlp->loss(labels, predicted) << endl;
		mlp->backward(labels, predicted, 0.1);
	}
	mlp->forward(inputs, predicted, false);
	cout<<"\t\tFinal Loss : "<< mlp->loss(labels, predicted) << endl;
	mlp->forward(inputs, predicted, test);

	cout << "\n\n\tXOR Testing : " << endl;
	for (int i = 0; i < 4; i++) {
		float maxValue = 0;
		int maxIndex = 0;
		cout << "\t\tFor ";
		for (int j = 0; j < 2; j++) {
			cout << "x" << j + 1 << " = " << inputs[i*3 + j]<<" ";
			
			if (predicted[i * 2 + j] > maxValue) {
				maxValue = predicted[i * 2 + j];
				maxIndex = j;
			}
		}
		cout << ", output = " << maxIndex << endl;
	}
	xorData.close();

	

	numEpochs = 100;

	int numberOfInstancesAlpha = 52;
	int numberOfFeaturesAlpha = 10201;
	int numberOfClassesAlpha = 52;
	float *input = new float[numberOfInstancesAlpha * numberOfFeaturesAlpha];
	float *true_labels = new float[numberOfInstancesAlpha * numberOfClassesAlpha];
	memset(true_labels, 0, numberOfInstancesAlpha * numberOfClassesAlpha * sizeof(float));
	for (int i = 0; i < numberOfInstancesAlpha; i++) {
		ifstream file("S:\\CIS 565\\Project_2\\Project2-Number-Algorithms\\Project2-Character-Recognition\\data-set\\" + ((i + 1 < 10) ? to_string(0) : "") + to_string(i + 1) + "info.txt");
		if (!file.is_open()) {
			exit(-1);
		}
		int count = 0;
		string line;
		while (getline(file, line))
		{
			count++;
			if (count == 1) {
				int index = i* numberOfClassesAlpha + (stof(line) - 1);
				true_labels[index] = 1;
			}
			if (count == 3) {
				stringstream ssin(line);
				for (int k = 0; ssin.good() && k < numberOfFeaturesAlpha; k++) {
					string temp;
					ssin >> temp;
					input[(i * numberOfFeaturesAlpha) + k] = stof(temp) / 255;
				}
			}
		}
		file.close();
	}


	int hiddenDimensionsAlpha[] = { 200 };
	PointCloudClassification::GraphConvolutionNetworkGPU *mlpAlpha = new PointCloudClassification::GraphConvolutionNetworkGPU(numberOfFeaturesAlpha, 1, hiddenDimensionsAlpha, numberOfClassesAlpha, numberOfInstancesAlpha);
	delete(predicted);
	predicted = new float[numberOfInstancesAlpha * numberOfClassesAlpha];




	printf("\n");
	printf("*************************\n");
	printf("* Character Recognition *\n");
	printf("*************************\n");
	cout << "\tCharacter Recognition Training : " << endl;
	ofstream characterData;
	characterData.open(R"(..\bookKeeping\characterLosses.csv)");
	for (int i = 0; i < numEpochs; i++) {
		mlpAlpha->forward(input, predicted);
		characterData << mlpAlpha->loss(true_labels, predicted) << endl;
		mlpAlpha->backward(true_labels, predicted, 0.01);
	}
	characterData.close();

	mlpAlpha->forward(input, predicted, false);
	cout << "\t\tFinal Loss : " << mlp->loss(true_labels, predicted) << endl;


	cout << "\n\tCharacter Recognition Testing : " << endl;
	srand(seed);
	for (int k = 0; k < 5; k++) {
		float maxTrueValue = 0;
		int maxTrueIndex = 0;
		float maxValue = 0;
		int maxIndex = 0;
		int i = rand() % numberOfInstancesAlpha;
		for (int j = 0; j < numberOfClassesAlpha; j++) {

			if (predicted[i * numberOfClassesAlpha + j] > maxValue) {
				maxValue = predicted[i * numberOfClassesAlpha + j];
				maxIndex = j;
			}
			if (true_labels[i * numberOfClassesAlpha + j] > maxTrueValue) {
				maxTrueValue = predicted[i * numberOfClassesAlpha + j];
				maxTrueIndex = j;
			}
		}
		cout << "\t\tFor True Label = " << (char)((maxTrueIndex % 2 == 0) ? 65 + (maxTrueIndex/2) : 97 + ((maxTrueIndex - 1)/2));
		cout << ", Predicted = " << (char)((maxIndex % 2 == 0) ? 65 + (maxIndex/2) : 97 + ((maxIndex - 1) / 2)) << endl;
	}


}
