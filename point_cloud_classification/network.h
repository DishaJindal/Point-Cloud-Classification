#pragma once

#include "common.h"
#include "hidden_layers/layer.h"
#include "hidden_layers/loss.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
    Common::PerformanceTimer& timer();

	class NetworkCPU {

		std::vector<Layer*> layers;
		Loss *loss;
		int batchSize;
		int numClasses;
		int inputFeatures; //Number of elements in every element of the vector (of batches)

	public :
		NetworkCPU(int inputFeatures, int numClasses, int batchSize);
		void addLayer(Layer* layer);

		std::vector<float*> forward(std::vector<float*> input, bool test);
		void backward(float *trueLabel, float *prediction, float learningRate);
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
