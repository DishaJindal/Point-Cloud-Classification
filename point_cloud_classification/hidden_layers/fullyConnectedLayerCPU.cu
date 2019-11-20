#pragma once

#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/matrix.h"
#include "../utilities/utils.h"
#include "layer.h"
#include "fullyConnectedLayer.h"
#include <fstream>
#include <string>


#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	class FullyConnectedLayerCPU : public FullyConnectedLayer {
	public:
		FullyConnectedLayerCPU() {};

		FullyConnectedLayerCPU(int inputDim, int outputDim, int batchDim, bool lastLayer) : FullyConnectedLayer(inputDim, outputDim, batchDim, lastLayer)  {

			// Randomly initialize weight matrix
			W = (float*)malloc(inputDim * outputDim * sizeof(float));
			Utilities::genArray(inputDim * outputDim, W);
		}

		/*
			outputArg = inputArg x W
		*/
		std::vector<float*> FullyConnectedLayer::forward(std::vector<float*> inputArg, bool test) {
			float* flattenedInput = (float*) malloc(batchDim * inputDim * sizeof(float));
			int i = 0;
			for(auto current: inputArg){
				memcpy(flattenedInput + (i * inputDim), current, inputDim * sizeof(float));
				i++;
			}
			/*std::cout << "Flattened: " << std::endl;
			std::cout << flattenedInput[(0 * inputDim) + 0] << " " << flattenedInput[(0 * inputDim) + 1] << " " << flattenedInput[(0 * inputDim) + 2] << std::endl;
			std::cout << flattenedInput[(1 * inputDim) + 0] << " " << flattenedInput[(1 * inputDim) + 1] << " " << flattenedInput[(1 * inputDim) + 2] << std::endl;
			std::cout << flattenedInput[(2 * inputDim) + 0] << " " << flattenedInput[(2 * inputDim) + 1] << " " << flattenedInput[(2 * inputDim) + 2] << std::endl;*/
			float* flattenedOutput = (float*) malloc(batchDim * outputDim * sizeof(float));
			
			MatrixCPU* m = new MatrixCPU();
			m->multiply(flattenedInput, W, batchDim, inputDim, outputDim, flattenedOutput);
			//free(flattenedInput);
			
			// Store input and output of this layer
			/*memcpy(A, flattenedInput, batchDim * inputDim * sizeof(float));
			memcpy(Z, flattenedOutput, batchDim * outputDim * sizeof(float));*/

			std::vector<float*> outputArg;
			for(int i = 0; i < batchDim; i++){
				outputArg.push_back(flattenedOutput + (i * outputDim));
			}
			//free(flattenedOutput);
			
			return outputArg;
		}

		/*
			outgoingGradient = incomingGradient x W.T
			dW = A.T x incomingGradient
		*/
		void FullyConnectedLayer::backward(float *incomingGradient, float *outgoingGradient, float learningRate) {
			MatrixCPU* m = new MatrixCPU();
			
			// Compute gradient w.r.t weights
			float *ATranspose = (float*) malloc(inputDim * batchDim * sizeof(float));
			m->transpose(A, batchDim, inputDim, ATranspose);
			m->multiply(ATranspose, incomingGradient, inputDim, batchDim, outputDim, dW);

			// Compute outgoingGradient (w.r.t. input)
			m->multiplyTranspose(incomingGradient, W, batchDim, outputDim, inputDim, outgoingGradient);

			//Update weight matrix
			m->subtractWithFactor(W, dW, learningRate, inputDim, outputDim, W);
		}
	};
}
