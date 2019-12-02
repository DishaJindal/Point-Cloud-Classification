#pragma once

#include "../common.h"
#include "layer.h"
#include "../utilities/utils.h"
#include <vector>
#include <math.h>

namespace PointCloudClassification {
	class GraphConvolutionLayerCPU : public Layer {
		protected : 
			/* 
				Weight matrix
			*/
			std::vector<float*> theta;
			/* 
				Derivative w.r.t. weight matrix
			*/

			std::vector<float*> X;
			std::vector<float*> L;

			int inputDim;
			int outputDim;

			int batchDim;
			int numPoints;
			int numFilters;

			bool lastLayer;
			Common::PerformanceTimer& timer()
			{
				static Common::PerformanceTimer timer;
				return timer;
			}


	public:
		GraphConvolutionLayerCPU() {};
		GraphConvolutionLayerCPU(int numPoints, int inputDim, int outputDim, int batchDim, int numFilters, bool lastLayer) {
			this->numPoints = numPoints;
			this->inputDim = inputDim;
			this->outputDim = outputDim;
			this->batchDim = batchDim;
			this->numFilters = numFilters;
			this->lastLayer = lastLayer;
			for (int i = 0; i < numFilters; i++) {
				float* temp = (float*)malloc(inputDim * outputDim * sizeof(float));
				Utilities::genArray(inputDim * outputDim, temp);
				theta.push_back(temp);
			}
		}

		int getInputDim() {
			return inputDim;
		}

		int getOutputDim() {
			return outputDim;
		}
		
		std::vector<float*> forward(std::vector<float*> input, bool test = false);
		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate);
		float* GraphConvolutionLayerCPU::get_chebeshev_polynomial(float* Tk1, float* Tk2, float* L, float mul = false);
	};

	class GraphConvolutionLayerGPU : public Layer {
	protected:
		/*
			Weight matrix
		*/
		std::vector<float*> theta;
		/*
			Derivative w.r.t. weight matrix
		*/

		std::vector<float*> X;
		std::vector<float*> L;

		int inputDim;
		int outputDim;

		int batchDim;
		int numPoints;
		int numFilters;

		bool lastLayer;

	public:
		GraphConvolutionLayerGPU() {};
		GraphConvolutionLayerGPU(int numPoints, int inputDim, int outputDim, int batchDim, int numFilters, bool lastLayer) {
			this->numPoints = numPoints;
			this->inputDim = inputDim;
			this->outputDim = outputDim;
			this->batchDim = batchDim;
			this->numFilters = numFilters;
			this->lastLayer = lastLayer;
			for (int i = 0; i < numFilters; i++) {
				float* temp;
				cudaMalloc((void**)&temp, inputDim * outputDim * sizeof(float));
				float* temp_cpu = (float*)malloc(inputDim * outputDim * sizeof(float));
				Utilities::genArray(inputDim * outputDim, temp_cpu);
				cudaMemcpy(temp, temp_cpu, inputDim * outputDim * sizeof(float), cudaMemcpyHostToDevice);
				theta.push_back(temp);
			}
		}

		int getInputDim() {
			return inputDim;
		}

		int getOutputDim() {
			return outputDim;
		}

		std::vector<float*> forward(std::vector<float*> input, bool test = false);
		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate);
		float* get_chebeshev_polynomial(float* Tk1, float* Tk2, float* L, float mul = false);
	};
}
