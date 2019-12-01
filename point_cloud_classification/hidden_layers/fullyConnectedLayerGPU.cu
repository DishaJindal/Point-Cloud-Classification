#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "../utilities/matrix.h"
#include "layer.h"
#include "fullyConnectedLayer.h"
#include <fstream>
#include <string>
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	class FullyConnectedLayerGPU : public FullyConnectedLayer {
		FullyConnectedLayerGPU() {};
		
	public:
		FullyConnectedLayerGPU(int inputDim, int outputDim, int batchDim, bool lastLayer) : FullyConnectedLayer(inputDim, outputDim, batchDim, lastLayer)  {
			cudaMalloc((void **)&W, inputDim * outputDim * sizeof(float));
			float *weightRand = new float[inputDim * outputDim];
			Utilities::genArray(inputDim * outputDim, weightRand);
			cudaMemcpy(W, weightRand, inputDim * outputDim * sizeof(float), cudaMemcpyHostToDevice);
			cudaMalloc((void**)&A, inputDim * batchDim * sizeof(float));
			cudaMalloc((void**)&dW, inputDim * outputDim * sizeof(float));
		}


		std::vector<float*> FullyConnectedLayer::forward(std::vector<float*> inputArg, bool test) {
			float* flattenedInput;
			cudaMalloc((void**)&flattenedInput, batchDim * inputDim * sizeof(float));
			int i = 0;
			for (auto current : inputArg) {
				cudaMemcpy(flattenedInput + (i * inputDim), current, inputDim * sizeof(float), cudaMemcpyDeviceToDevice);
				i++;
			}
			float* flattenedOutput;
			cudaMalloc((void**)&flattenedOutput, batchDim * outputDim * sizeof(float));

			MatrixGPU* m = new MatrixGPU();
			m->multiply(flattenedInput, W, batchDim, inputDim, outputDim, flattenedOutput);
			//free(flattenedInput);

			// Store input and output of this layer
			cudaMemcpy(A, flattenedInput, batchDim * inputDim * sizeof(float), cudaMemcpyDeviceToDevice);

			std::vector<float*> outputArg;
			for (int i = 0; i < batchDim; i++) {
				outputArg.push_back(flattenedOutput + (i * outputDim));
			}
			//free(flattenedOutput);

			return outputArg;
		}

		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate) {
			float* flattenedInput;
			cudaMalloc((void**)&flattenedInput, batchDim * outputDim * sizeof(float));
			int i = 0;
			for (auto current : incomingGradient) {
				cudaMemcpy(flattenedInput + (i * outputDim), current, outputDim * sizeof(float), cudaMemcpyDeviceToDevice);
				i++;
			}
			float* flattenedOutput;
			cudaMalloc((void**)&flattenedOutput, batchDim * inputDim * sizeof(float));

			MatrixGPU* m = new MatrixGPU();

			// Compute gradient w.r.t weights
			float *ATranspose;
			cudaMalloc((void**)&ATranspose, inputDim * batchDim * sizeof(float));
			m->transpose(A, batchDim, inputDim, ATranspose);
			m->multiply(ATranspose, flattenedInput, inputDim, batchDim, outputDim, dW);

			// Compute outgoingGradient (w.r.t. input)
			m->multiplyTranspose(flattenedInput, W, batchDim, outputDim, inputDim, flattenedOutput);

			//Update weight matrix
			m->subtractWithFactor(W, dW, learningRate, inputDim, outputDim, W);
			
			std::vector<float*> outgoingGradient;
			for (int i = 0; i < batchDim; i++) {
				outgoingGradient.push_back(flattenedOutput + (i * inputDim));
			}
			//free(flattenedOutput);

			return outgoingGradient;

		}
	};
}
