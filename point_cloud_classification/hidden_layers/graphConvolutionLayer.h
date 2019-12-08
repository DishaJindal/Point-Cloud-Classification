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

		float* Tk;
		float* Tk_minus_2;
		float* Tk_minus_1;
		float* current_output;
		std::vector<float*> output;

		float* TX;
		float* TXT;
		float* dtheta;
		float* Tk_back;
		float* Tk_minus_2_back;
		float* Tk_minus_1_back;
		float* TG;
		float* temp;
		float* thetaT;
		float* current_outgoing_gradient;
		std::vector<float*> outgoing_gradient;

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
			cudaMalloc((void**)&Tk, numPoints * inputDim * sizeof(float));

			cudaMalloc((void**)&Tk_minus_2, numPoints * inputDim * sizeof(float));

			cudaMalloc((void**)&Tk_minus_1, numPoints * inputDim * sizeof(float));

			for (int i = 0; i < batchDim; i++) {
				cudaMalloc((void**)&current_output, numPoints * outputDim * sizeof(float));
				output.push_back(current_output);
			}

			// Backward mallocs

			cudaMalloc((void**)&TX, numPoints * inputDim * sizeof(float));
			//checkCUDAError("cudaMalloc((void**)&TX");

			cudaMalloc((void**)&TXT, numPoints * inputDim * sizeof(float));
			//checkCUDAError("cudaMalloc((void**)&TXT");

			cudaMalloc((void**)&dtheta, inputDim * outputDim * sizeof(float));
			//checkCUDAError("cudaMalloc((void**)&dtheta");

			cudaMalloc((void**)&Tk_back, numPoints * numPoints * sizeof(float));

			cudaMalloc((void**)&Tk_minus_2_back, numPoints * numPoints * sizeof(float));
			//checkCUDAError("cudaMalloc((void**)&Tk_minus_2");

			cudaMalloc((void**)&Tk_minus_1_back, numPoints * numPoints * sizeof(float));
			//checkCUDAError("cudaMalloc((void**)&Tk_minus_1");

			cudaMalloc((void**)&TG, numPoints * outputDim * sizeof(float));
			//checkCUDAError("cudaMalloc((void**)&TG");

			cudaMalloc((void**)&temp, numPoints * inputDim * sizeof(float));
			//checkCUDAError("cudaMalloc((void**)&temp");

			cudaMalloc((void**)&thetaT, outputDim * inputDim * sizeof(float));
			//checkCUDAError("cudaMalloc((void**)&thetaT");

			for (int i = 0; i < batchDim; i++) {
				cudaMalloc((void**)&current_outgoing_gradient, numPoints * inputDim * sizeof(float));
				outgoing_gradient.push_back(current_outgoing_gradient);
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
		void get_chebeshev_polynomial(float* Tk1, float* Tk2, float* L, float* Tk, float mul = false);
	};
}
