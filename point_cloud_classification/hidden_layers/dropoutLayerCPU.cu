#pragma once

#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/matrix.h"
#include "../utilities/utils.h"
#include "../utilities/parameters.h"
#include "layer.h"
#include "dropoutLayer.h"
#include <fstream>
#include <string>
#include <random>

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	//class DropoutLayerCPU : public DropoutLayer {
	//public:
	//	DropoutLayerCPU() {};

	//	DropoutLayerCPU(int numPoints, int inputDim, int batchDim, bool lastLayer, float k_prob) : DropoutLayer(numPoints, inputDim, batchDim, lastLayer) {
	//		// Allocate space to save which nodes were kept required for back propagataion
	//		for (int i = 0; i < batchDim; i++)
	//			this->nodesKeep.push_back((float*)malloc(inputDim * numPoints * sizeof(float)));
	//		std::default_random_engine generator;
	//		std::bernoulli_distribution distribution(k_prob);
	//		this->rand_generator = generator;
	//		this->disturb = distribution;
	//	}

		/*
			inputArg -> batchDim x inputDim
			outputArg -> batchDim x inputDim
		*/
		std::vector<float*> DropoutLayerCPU::forward(std::vector<float*> inputArg, bool test) {
			std::vector<float*> output;
			for (int b = 0; b < batchDim; b++) {
				float* current_output = (float*)malloc(numPoints * inputDim * sizeof(float));
				for (int i = 0; i < numPoints * inputDim; i++) {
					// Zero out nodes and save which ones are zeroed out
					this->nodesKeep[b][i] = (this->disturb(this->rand_generator)) / Parameters::keep_prob; 
					current_output[i] = inputArg[b][i] * this->nodesKeep[b][i];
				}
				output.push_back(current_output);
			}
			return output;
		}

		std::vector<float*> DropoutLayerCPU::backward(std::vector<float*> incomingGradient, float learningRate) {
			for (int b = 0; b < batchDim; b++) {
				for (int i = 0; i < numPoints * inputDim; i++) {
					incomingGradient[b][i] *= this->nodesKeep[b][i]; // Does inplace gradient update
				}
			}
			return incomingGradient;
		}
	};
