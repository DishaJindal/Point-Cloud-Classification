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

//__global__ void kernExp(float* input, int m, int n, float* output) {
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//	if (index < m * n) {
//		output[index] = exp(input[index]);
//	}
//}
//
//__global__ void kernDivideSum(float* input, float* sum, int m, int n, float* output) {
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//	int row = index / n;
//	if (index < m * n) {
//		output[index] = input[index] / sum[row];
//	}
//}

namespace PointCloudClassification {

	class softmaxActivationLayerGPU : public softmaxActivationLayer {
		softmaxActivationLayerGPU() {};
	public:
		softmaxActivationLayerGPU(int inputDim, int batchDim, bool lastLayer) : softmaxActivationLayer(inputDim, inputDim, batchDim, lastLayer) {
			
		}

		std::vector<float*> forward(std::vector<float*> inputArg, bool test) {
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

			dim3 fullBlocksPerGrid((batchDim * inputDim + blockSize - 1) / blockSize);
			//kernExp <<<fullBlocksPerGrid, blockSize >>> (flattenedInput, batchDim, inputDim, temp);

			MatrixGPU* m = new MatrixGPU();
			float* tempT;
			cudaMalloc((void**)&tempT, batchDim * outputDim * sizeof(float));
			m->transpose(temp, batchDim, outputDim, tempT);
			//cudaFree(temp);

			float* sum;
			cudaMalloc((void**)&sum, batchDim * outputDim * sizeof(float));
			m->meanAcrossDim1(tempT, outputDim, batchDim, sum); //CHANGE THIS TO SUM --> m->sumAcrossDim1(tempT, outputDim, batchDim, sum);

			//dim3 fullBlocksPerGrid((batchDim * inputDim + blockSize - 1) / blockSize);
			//kernDivideSum << <fullBlocksPerGrid, blockSize >> > (temp, sum, batchDim, inputDim, flattenedOutput);

			cudaFree(temp);
			cudaFree(tempT);
			cudaFree(sum);

			std::vector<float*> outputArg;
			for (int i = 0; i < batchDim; i++) {
				outputArg.push_back(flattenedOutput + (i * outputDim));
			}

			return outputArg;
		}

		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate) {
			std::vector<float*> sample;
			return sample;
		}
	};
}
