#pragma once

#include "common.h"
#include "hidden_layers/layer.h"
#include "hidden_layers/loss.h"
#include <vector>
#include <math.h>
#include "hidden_layers/fullyConnectedLayer.h"
#include "hidden_layers/globalPoolingLayer.h"
#include "hidden_layers/RELUActivationLayer.h"
#include "hidden_layers/dropoutLayer.h"
#include "hidden_layers/graphConvolutionLayer.h"
#include "hidden_layers/softmaxActivationLayer.h"

namespace PointCloudClassification {
    Common::PerformanceTimer& timer();

	class NetworkCPU {

		std::vector<Layer*> layers;
		Loss *loss;
		int batchSize;
		int numClasses;
		Layer* softmaxFunction;

		// Architecture Layers
		PointCloudClassification::GraphConvolutionLayerCPU gcn_layer1;
		PointCloudClassification::DropoutLayerCPU dropout_layer1;
		PointCloudClassification::GlobalPoolingLayerCPU gp_layer1;
		PointCloudClassification::GraphConvolutionLayerCPU gcn_layer2;
		PointCloudClassification::DropoutLayerCPU dropout_layer2;
		PointCloudClassification::GlobalPoolingLayerCPU gp_layer2;
		PointCloudClassification::DropoutLayerCPU dropout_layer3;
		PointCloudClassification::FullyConnectedLayerCPU fc_layer1;
		PointCloudClassification::RELUActivationLayerCPU relu1;
		PointCloudClassification::DropoutLayerCPU dropout_layer4;
		PointCloudClassification::FullyConnectedLayerCPU fc_layer2;

	public :
		NetworkCPU(int numClasses, int batchSize);
		void addLayer(Layer* layer);
		void setLoss(Loss* loss);
		void buildArchitecture();
		std::vector<float*> forward(std::vector<float*> input, bool test);
		float calculateLoss(std::vector<float*> prediction, std::vector<float*> trueLabel);
		void backward(std::vector<float*> prediction, std::vector<float*> trueLabel, float learningRate);
		void train(std::vector<float*> input, std::vector<float*> label, int n);
		void getClassification(const std::vector<float*> prediction, const int classes, std::vector<float> classification);
	};

	class NetworkGPU {

		std::vector<Layer*> layers;
		Loss *loss;
		int batchSize;
		int numClasses;
		Layer* softmaxFunction;

		// Architecture Layers
		PointCloudClassification::GraphConvolutionLayerGPU gcn_layer1;
		PointCloudClassification::DropoutLayerGPU dropout_layer1;
		PointCloudClassification::GlobalPoolingLayerGPU gp_layer1;
		PointCloudClassification::GraphConvolutionLayerGPU gcn_layer2;
		PointCloudClassification::DropoutLayerGPU dropout_layer2;
		PointCloudClassification::GlobalPoolingLayerGPU gp_layer2;
		PointCloudClassification::DropoutLayerGPU dropout_layer3;
		PointCloudClassification::FullyConnectedLayerGPU fc_layer1;
		PointCloudClassification::RELUActivationLayerGPU relu1;
		PointCloudClassification::DropoutLayerGPU dropout_layer4;
		PointCloudClassification::FullyConnectedLayerGPU fc_layer2;
	public:
		NetworkGPU(int numClasses, int batchSize);
		void addLayer(Layer* layer);
		void setLoss(Loss* loss);
		void buildArchitecture();
		std::vector<float*> forward(std::vector<float*> input, bool test);
		float calculateLoss(std::vector<float*> prediction, std::vector<float*> trueLabel);
		void backward(std::vector<float*> prediction, std::vector<float*> trueLabel, float learningRate);
		void train(std::vector<float*> input, std::vector<float*> label, int n);
		void getClassification(const std::vector<float*> prediction, const int classes, std::vector<float> classification);
	};
}
