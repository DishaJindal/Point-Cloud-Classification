#pragma once

#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "layer.h"
#include "globalPoolingLayer.h"
#include <fstream>
#include <string>
#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	/*
		inputArg -> N x D (N - number of points per sample)
		outputArg -> 1 x D
		Takes maximum across all points
	*/
	std::vector<float*> GlobalPoolingLayerCPU::forward(std::vector<float*> inputArg, bool test) {
		// Save X for back prop
		this->Z = inputArg;

		// Calculate Output
		std::vector<float*> output;
		for (int b = 0; b < inputArg.size(); b++) {
			float* current_output = (float*)malloc(inputDim * 2 * sizeof(float));
				
			// Max Pooling and Save Argmax for back prop
			m->maxAcrossDim1(inputArg[b], numPoints, inputDim, this->argMax[b], current_output);

			// Calculate mean and Save it for back prop
			m->meanAcrossDim1(inputArg[b], numPoints, inputDim, this->mean[b]);
				
			// Variance Pooling
			m->varianceAcrossDim1(inputArg[b], numPoints, inputDim, current_output + inputDim, this->mean[b]);
			output.push_back(current_output);
		}
		return output;
	}

	/*
		incomingGradient -> B*D*2
		returns -> B*N*D
		
	*/
	std::vector<float*> GlobalPoolingLayerCPU::backward(std::vector<float*> incomingGradient, float learningRate) {
		std::vector<float*> outgoingGradient;
		for (int b = 0; b < this->batchDim; b++) {
			float* oneOutgoingGradient = (float*)malloc(numPoints * inputDim * sizeof(float));

			// Consume Gradient coming from max pooling
			for (int d = 0; d < this->inputDim; d++) {
				// Initialize gradients propagation to all points with 0
				for (int n = 0; n < this->numPoints; n++) {
					oneOutgoingGradient[this->inputDim * n + d] = 0;
				}
				// Update gradient of the max point
				oneOutgoingGradient[this->inputDim * this->argMax[b][d] + d] = incomingGradient[b][d];
			}
				
			// Consume Gradient coming from variance pooling
			for (int d = 0; d < this->inputDim; d++) {
				for (int n = 0; n < this->numPoints; n++) {
					float del_yj_by_xij = (2 * ((Z[b][n* inputDim + d]) - this->mean[b][d])) / numPoints;
					oneOutgoingGradient[this->inputDim * n + d] += incomingGradient[b][d] * del_yj_by_xij;
				}
			}
			outgoingGradient.push_back(oneOutgoingGradient);
		}
		return outgoingGradient;
	}

};

