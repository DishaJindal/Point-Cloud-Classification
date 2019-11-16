#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "network.h"
#include "layers/layer.h"
#include "layers/implementations/fullyConnectedLayerGPU.cu"
#include "layers/fullyConnectedLayer.h"
#include <fstream>
#include <string>
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

using namespace Utilities;

namespace PointCloudClassification {

	float GraphConvolutionNetworkGPU::loss(float *label, float *predicted) {

		float *devLabel;
		cudaMalloc((void**)&devLabel, batchDim*layers[layers.size() - 1]->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(devLabel, label, batchDim * layers[layers.size() - 1]->getOutputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");

		float *devPredicted;
		cudaMalloc((void**)&devPredicted, batchDim*layers[layers.size() - 1]->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(devPredicted, predicted, batchDim*layers[layers.size() - 1]->getOutputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");
		

		float *lossForEachLabel = new float[batchDim * layers[layers.size() - 1]->getOutputDim()];
		float *devLossForEachLabel;
		cudaMalloc((void**)&devLossForEachLabel, batchDim * layers[layers.size() - 1]->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		

		int gridRows = ((batchDim * layers[layers.size() - 1]->getOutputDim()) + blockSize - 1) / blockSize;
		kernCrossEntropyLoss << <gridRows, blockSize >> > (batchDim * layers[layers.size() - 1]->getOutputDim(), devPredicted, devLabel, devLossForEachLabel);
		checkCUDAError("kernCrossEntropyLoss");

		cudaMemcpy(lossForEachLabel, devLossForEachLabel, batchDim * layers[layers.size() - 1]->getOutputDim() * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy");

		float loss = 0;

		for (int i = 0; i < batchDim * layers[layers.size() - 1]->getOutputDim(); i++) {
			loss += lossForEachLabel[i];
		}
		return loss / batchDim;
		
	}

	GraphConvolutionNetworkGPU::GraphConvolutionNetworkGPU(int inputDim, int numHiddenLayers, int *hiddenDim, int outputDim, int batchDim) {
		this->batchDim = batchDim;

		FullyConnectedLayer *tempLayer = new FullyConnectedLayerGPU(inputDim, hiddenDim[0], batchDim, false);
		layers.push_back(tempLayer);
		for (int i = 1; i < numHiddenLayers - 1; i++) {
			FullyConnectedLayerGPU *tempLayer = new FullyConnectedLayerGPU(hiddenDim[i - 1], hiddenDim[i], batchDim, false);
			layers.push_back(tempLayer);
		}
		tempLayer = new FullyConnectedLayerGPU(hiddenDim[numHiddenLayers - 1], outputDim, batchDim, true);
		layers.push_back(tempLayer);

	}

	void GraphConvolutionNetworkGPU::forward(float *input, float *output, bool test) {
		float *devOutput;
		cudaMalloc((void**)&devOutput, batchDim * layers[0]->getInputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(devOutput, input, batchDim * layers[0]->getInputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");
		float *hiddenOutput;
		for (int i = 0; i < layers.size(); i++) {
			cudaMalloc((void**)&hiddenOutput, batchDim * layers[i]->getOutputDim() * sizeof(float));
			checkCUDAError("cudaMalloc");
			layers[i]->forward(devOutput, hiddenOutput, test);
			cudaFree(devOutput);
			cudaMalloc((void**)&devOutput, batchDim * layers[i]->getOutputDim() * sizeof(float));
			checkCUDAError("cudaMalloc");
			cudaMemcpy(devOutput, hiddenOutput, batchDim * layers[i]->getOutputDim() * sizeof(float), cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy");
			cudaFree(hiddenOutput);
		}
		cudaMemcpy(output, devOutput, batchDim * layers[layers.size() - 1]->getOutputDim() * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy");

		cudaFree(devOutput);
	}

	void GraphConvolutionNetworkGPU::backward(float *label, float *predicted, float learningRate) {
		float *devLabel;
		cudaMalloc((void**)&devLabel, batchDim*layers[layers.size() - 1]->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(devLabel, label, batchDim * layers[layers.size() - 1]->getOutputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");

		float *devPredicted;
		cudaMalloc((void**)&devPredicted, batchDim*layers[layers.size() - 1]->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(devPredicted, predicted, batchDim*layers[layers.size() - 1]->getOutputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");

		float *incomingGradient;
		cudaMalloc((void**)&incomingGradient, batchDim*layers[layers.size() - 1]->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");


		int gridRows = ((batchDim * layers[layers.size() - 1]->getOutputDim()) + blockSize - 1) / blockSize;
		kernSubtractMatrices << <gridRows, blockSize >> > (devPredicted, devLabel, incomingGradient, batchDim, layers[layers.size() - 1]->getOutputDim());
		checkCUDAError("kernSubtractMatrices");

		cudaFree(devLabel);
		checkCUDAError("cudaFree");
		cudaFree(devPredicted);
		checkCUDAError("cudaFree");


		checkCUDAError("cudaMemcpy");
		float *outgoingGradient;
		for (int i = layers.size() - 1; i >= 0; i--) {
			cudaMalloc((void**)&outgoingGradient, batchDim*layers[i]->getInputDim() * sizeof(float));
			checkCUDAError("cudaMalloc");
			layers[i]->backward(learningRate, incomingGradient, outgoingGradient);
			cudaFree(incomingGradient);
			checkCUDAError("cudaFree");
			cudaMalloc((void**)&incomingGradient, batchDim*layers[i]->getInputDim() * sizeof(float));
			checkCUDAError("cudaMalloc");
			cudaMemcpy(incomingGradient, outgoingGradient, batchDim*layers[i]->getInputDim() * sizeof(float), cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy");
			cudaFree(outgoingGradient);
		}
		cudaFree(incomingGradient);
	}


}
