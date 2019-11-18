#pragma once

#include "common.h"
#include "hidden_layers/layer.h"
#include "hidden_layers/loss.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
    Common::PerformanceTimer& timer();

	class GraphConvolutionNetworkCPU {

		std::vector<Layer*> layers;
		Loss *loss;
		int batchDim;
	public :
		GraphConvolutionNetworkCPU(int inputDim, int outputDim, int batchDim);
		void addLayer(Layer* layer);

		void forward(float *input, float *prediction, bool test = false);
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
