#pragma once

#include "../common.h"
#include "layer.h"
#include <vector>
#include <math.h>
#include <random>

namespace PointCloudClassification {
	class DropoutLayer : public Layer {
	protected:
		/*
			Input
		*/
		std::vector<float *> Z;

		int numPoints;
		int inputDim;
		int batchDim;
		bool lastLayer;
		std::vector<float*> nodesKeep;
		std::bernoulli_distribution disturb; 
		std::default_random_engine rand_generator;

	public:
		DropoutLayer() {};
		DropoutLayer(int numPoints, int inputDim, int batchDim, bool lastLayer) {
			this->numPoints = numPoints; 
			this->inputDim = inputDim;
			this->batchDim = batchDim;
			this->lastLayer = lastLayer;
		}

		int getInputDim() {
			return inputDim;
		}

		int getOutputDim() {
			return inputDim;
		}

		std::vector<float*> forward(std::vector<float*> input, bool test = false) = 0;
		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate) = 0;
	};
}