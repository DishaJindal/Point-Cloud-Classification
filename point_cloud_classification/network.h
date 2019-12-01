#pragma once

#include "common.h"
#include "hidden_layers/layer.h"
#include "hidden_layers/loss.h"
#include <vector>
#include <math.h>
#include "hidden_layers/fullyConnectedLayerCPU.cu"
#include "hidden_layers/softmaxActivationLayerCPU.cu"
#include "hidden_layers/graphConvolutionLayerCPU.cu"
#include "hidden_layers/dropoutLayerCPU.cu"
#include "hidden_layers/globalPoolingCPU.cu"
#include "hidden_layers/RELUActivationLayerCPU.cu"

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
		void train(std::vector<float*> input, std::vector<float*> laplacians, std::vector<float*> label, int n);
		void getClassification(const std::vector<float*> prediction, const int classes, std::vector<float> classification);
	};

	class GraphConvolutionNetworkGPU {

		std::vector<Layer*> layers;
		int batchDim;
	public:
		GraphConvolutionNetworkGPU(int inputDim, int numHiddenLayers, int *hiddenDim, int outputDim, int batchDim);
		void forward(float *input, float *output, bool test = false);
		void backward(float *output, float *predicted, float learningRate);
		float loss(float *label, float *predicted);
	};
}
