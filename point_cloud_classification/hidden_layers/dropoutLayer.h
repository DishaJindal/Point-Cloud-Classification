#pragma once

#include "../common.h"
#include "layer.h"
#include <vector>
#include <math.h>
#include <random>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

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


	class DropoutLayerCPU : public DropoutLayer {
	public:
		DropoutLayerCPU() {}
		DropoutLayerCPU(int numPoints, int inputDim, int batchDim, bool lastLayer, float k_prob) : DropoutLayer(numPoints, inputDim, batchDim, lastLayer) {
			// Allocate space to save which nodes were kept required for back propagataion
			for (int i = 0; i < batchDim; i++)
				this->nodesKeep.push_back((float*)malloc(inputDim * numPoints * sizeof(float)));
			std::default_random_engine generator;
			std::bernoulli_distribution distribution(k_prob);
			this->rand_generator = generator;
			this->disturb = distribution;
		}
		std::vector<float*> forward(std::vector<float*> input, bool test = false);
		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate);
	};

	class DropoutLayerGPU : public DropoutLayer {
	public:
		DropoutLayerGPU() {}
		DropoutLayerGPU(int numPoints, int inputDim, int batchDim, bool lastLayer, float k_prob) : DropoutLayer(numPoints, inputDim, batchDim, lastLayer) {
			// Allocate space to save which nodes were kept required for back propagataion
			for (int i = 0; i < batchDim; i++) {
				float* flattened_current_random_numbers;
				cudaMalloc((void**)&flattened_current_random_numbers, numPoints * inputDim * sizeof(float));
				this->nodesKeep.push_back(flattened_current_random_numbers);
			}
			std::default_random_engine generator;
			this->rand_generator = generator;
		}
		std::vector<float*> forward(std::vector<float*> input, bool test = false);
		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate);
	};
}