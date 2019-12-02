#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "../utilities/matrix.h"
#include "layer.h"
#include "softmaxActivationLayer.h"
#include <fstream>
#include <string>
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128



namespace PointCloudClassification {

	std::vector<float*> softmaxActivationLayerGPU::forward(std::vector<float*> inputArg, bool test) {
		float* flattenedInput;
		cudaMalloc((void**)&flattenedInput, batchDim * inputDim * sizeof(float));
		int i = 0;
		for (auto current : inputArg) {
			cudaMemcpy(flattenedInput + (i * inputDim), current, inputDim * sizeof(float), cudaMemcpyDeviceToDevice);
			i++;
		}
		float* flattenedOutput;
		cudaMalloc((void**)&flattenedOutput, batchDim * outputDim * sizeof(float));

		float* temp;
		cudaMalloc((void**)&temp, batchDim * outputDim * sizeof(float));

		MatrixGPU* m = new MatrixGPU();
		m->exp(flattenedInput, batchDim, inputDim, temp);

		Utilities::printArrayGPU(temp, 10);

		float* tempT;
		cudaMalloc((void**)&tempT, batchDim * outputDim * sizeof(float));
		m->transpose(temp, batchDim, outputDim, tempT);
		//cudaFree(temp);

		//m->linearCombination(tempT, tempT, 1000, 0, outputDim, batchDim, tempT);

		float* sum;
		cudaMalloc((void**)&sum, batchDim * outputDim * sizeof(float));
		m->sumAcrossDim1(tempT, outputDim, batchDim, sum); //CHANGE THIS TO SUM --> m->sumAcrossDim1(tempT, outputDim, batchDim, sum);

		//m->linearCombination(sum, sum, 1.0f / 1000, 0, batchDim, 1, sum);

		//dim3 fullBlocksPerGrid((batchDim * inputDim + blockSize - 1) / blockSize);
		m->divide_sum (temp, sum, batchDim, outputDim, flattenedOutput);

		cudaFree(temp);
		cudaFree(tempT);
		cudaFree(sum);

		std::vector<float*> outputArg;
		for (int i = 0; i < batchDim; i++) {
			outputArg.push_back(flattenedOutput + (i * outputDim));
		}

		return outputArg;
	}

	std::vector<float*> softmaxActivationLayerGPU::backward(std::vector<float*> incomingGradient, float learningRate) {
		std::vector<float*> sample;
		return sample;
	}

};

