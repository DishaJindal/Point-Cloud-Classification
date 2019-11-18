#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
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
		}


		void forward(float *inputArg, float *outputArg, bool test) {
			cudaMemcpy(A, inputArg, batchDim * inputDim * sizeof(float), cudaMemcpyDeviceToDevice);
			int gridRows = (batchDim*outputDim + blockSize - 1) / blockSize;

			Utilities::kernMultiplyMatrices<<<gridRows, blockSize >>>(inputArg, W, outputArg, batchDim, inputDim, outputDim);
			checkCUDAError("kernMultiplyMatrices");

			dim3 fullBlocksPerGrid((outputDim*batchDim + blockSize - 1) / blockSize);
			if (!lastLayer) {
				Utilities::kernActivateReLU << <fullBlocksPerGrid, blockSize >> > (outputArg, outputDim*batchDim);
			}
			else {
				float *output = new float[outputDim * batchDim];
				cudaMemcpy(output, outputArg, batchDim * outputDim * sizeof(float), cudaMemcpyDeviceToHost);
				float *softmaxDenominator = new float[batchDim];
				memset(softmaxDenominator, 0, batchDim * sizeof(float));
				for (int j = 0; j < batchDim; j++) {
					for (int i = 0; i < outputDim; i++) {
						softmaxDenominator[j] += exp(output[j * outputDim + i]);
					}
				}

				float *devSoftmaxDenominator;
				cudaMalloc((void **)&devSoftmaxDenominator, batchDim * sizeof(float));
				cudaMemcpy(devSoftmaxDenominator, softmaxDenominator, batchDim * sizeof(float), cudaMemcpyHostToDevice);
				Utilities::kernActivateSoftmax << <fullBlocksPerGrid, blockSize >> > (outputArg, batchDim * outputDim, outputDim, devSoftmaxDenominator);
				checkCUDAError("kernActivateSoftmax");

				delete(output);

				cudaFree(devSoftmaxDenominator);
				checkCUDAError("cudaFree");
			}

			if (test) {
				printf("\n\n\tWeights : ");
				float *tempWeight = new float[inputDim * outputDim];
				cudaMemcpy(tempWeight, W, inputDim * outputDim * sizeof(float), cudaMemcpyDeviceToHost);
				for (int i = 0; i < inputDim * outputDim; i++) {
					if (i % outputDim == 0) {
						printf("\n\t\t");
					}
					printf("%f ", tempWeight[i]);
				}
				delete(tempWeight);
			}
		}

		void backward(float *incomingGradient, float *outgoingGradient, float learningRate) {

			float *weightTranspose;
			cudaMalloc((void**)&weightTranspose, inputDim * outputDim * sizeof(float));
			checkCUDAError("cudaMalloc");

			int gridRows = (inputDim*outputDim + blockSize - 1) / blockSize;
			Utilities::kernTransposeMatrices << <gridRows, blockSize >> > (W, weightTranspose, inputDim, outputDim);
			checkCUDAError("kernTransposeMatrices");

			float *outgoingGradientLocal;
			cudaMalloc((void**)&outgoingGradientLocal, inputDim*batchDim * sizeof(float));
			checkCUDAError("cudaMalloc");


			gridRows = (inputDim*batchDim + blockSize - 1) / blockSize;
			Utilities::kernMultiplyMatrices << <gridRows, blockSize >> > (incomingGradient, weightTranspose, outgoingGradientLocal, batchDim, outputDim, inputDim);
			checkCUDAError("kernMultiplyMatrices");

			cudaFree(weightTranspose);
			checkCUDAError("cudaFree");

			float *inputDerivatived;
			cudaMalloc((void**)&inputDerivatived, batchDim * inputDim * sizeof(float));
			dim3 fullBlocksPerGrid((inputDim * batchDim + blockSize - 1) / blockSize);
			Utilities::kernActivateReLUDerivative << <fullBlocksPerGrid, blockSize >> > (A, inputDerivatived, inputDim * batchDim);
			checkCUDAError("kernActivateReLUDerivative");


			gridRows = (inputDim*batchDim + blockSize - 1) / blockSize;
			Utilities::kernMultMatricesHammard << <gridRows, blockSize >> > (outgoingGradientLocal, inputDerivatived, outgoingGradient, batchDim, inputDim);
			checkCUDAError("kernMultMatricesHammard");

			cudaFree(inputDerivatived);
			checkCUDAError("cudaFree");


			float *inputTranspose;
			cudaMalloc((void**)&inputTranspose, inputDim * batchDim * sizeof(float));

			gridRows = (inputDim*batchDim + blockSize - 1) / blockSize;
			Utilities::kernTransposeMatrices << <gridRows, blockSize >> > (A, inputTranspose, batchDim, inputDim);
			checkCUDAError("kernTransposeMatrices");

			float *gradient;
			cudaMalloc((void**)&gradient, inputDim * outputDim * sizeof(float));
			gridRows = (inputDim*outputDim + blockSize - 1) / blockSize;
			Utilities::kernMultiplyMatrices << <gridRows, blockSize >> > (inputTranspose, incomingGradient, gradient, inputDim, batchDim, outputDim);
			checkCUDAError("kernMultiplyMatrices");

			cudaFree(inputTranspose);
			checkCUDAError("cudaFree");


			Utilities::kernMultMatricesWithScalar << <gridRows, blockSize >> > (gradient, gradient, inputDim, outputDim, learningRate);
			checkCUDAError("kernMultMatricesWithScalar");


			Utilities::kernSubtractMatrices << <gridRows, blockSize >> > (W, gradient, W, inputDim, outputDim);
			checkCUDAError("kernSubtractMatrices");

			cudaFree(gradient);
			checkCUDAError("cudaFree");
		}
	};
}
