#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/matrix.h"
#include "../utilities/parameters.h"
#include "layer.h"
#include "graphConvolutionLayer.h"
#include <fstream>
#include <string>
#include "device_launch_parameters.h"
#include <chrono>
#include <fstream>
using namespace std;

using namespace std::chrono;

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	void GraphConvolutionLayerGPU::get_chebeshev_polynomial(float* Tk1, float* Tk2, float* L, float*Tk, float mul) {
		MatrixGPU* m = new MatrixGPU();

		if (mul) {
			float* t;
			cudaMalloc((void**)&t, numPoints * inputDim * sizeof(float));
			m->multiply(L, Tk1, numPoints, numPoints, inputDim, t);

			m->linearCombination(t, Tk2, 2, -1, numPoints, inputDim, Tk);
			cudaFree(t);
		}
		else {
			float* t;
			cudaMalloc((void**)&t, numPoints * numPoints * sizeof(float));
			m->multiply(L, Tk1, numPoints, numPoints, numPoints, t);

			m->linearCombination(t, Tk2, 2, -1, numPoints, numPoints, Tk);
			cudaFree(t);
		}
	}

	void GraphConvolutionLayerGPU::saveModel(std::string file_name) {
		// Save Weights
		for (int i = 0; i < numFilters; i++) {
			float* W_host = new float[inputDim * outputDim];
			cudaMemcpy(W_host, theta[i], inputDim * outputDim * sizeof(float), cudaMemcpyDeviceToHost);
			ofstream out(file_name + std::to_string(i) + "_W.txt");
			for (int i = 0; i < inputDim * outputDim; ++i) {
				out << W_host[i] << "\n";
			}
			out.close();
		}
		//Utilities::printArrayGPU(theta[0], 10);
	}

	std::vector<float*> GraphConvolutionLayerGPU::forward(std::vector<float*> inputArg, bool test) {
		MatrixGPU* m = new MatrixGPU();
		this->X = std::vector < float* >(inputArg.begin(), inputArg.begin() + batchDim);
		this->L = std::vector < float* >(inputArg.begin() + batchDim, inputArg.end());

		for (int i = 0; i < batchDim; i++) {
			
			float* current_input = inputArg[i];
			float* current_L = inputArg[i + batchDim];

			//Tk_minus_2 = current_input;
			cudaMemcpy(Tk_minus_2, current_input, numPoints * inputDim * sizeof(float), cudaMemcpyDeviceToDevice);

			m->multiply(current_L, current_input, numPoints, numPoints, inputDim, Tk_minus_1);
			for (int k = 0; k < numFilters; k++) {
				if (k == 0) {
					//Tk = Tk_minus_2;
					cudaMemcpy(Tk, Tk_minus_2, numPoints * inputDim * sizeof(float), cudaMemcpyDeviceToDevice);
				}
				else if (k == 1) {
					//Tk = Tk_minus_1;
					cudaMemcpy(Tk, Tk_minus_1, numPoints * inputDim * sizeof(float), cudaMemcpyDeviceToDevice);
				}
				else {
					get_chebeshev_polynomial(Tk_minus_1, Tk_minus_2, current_L, Tk, true);
					//Tk_minus_2 = Tk_minus_1;
					cudaMemcpy(Tk_minus_2, Tk_minus_1, numPoints * inputDim * sizeof(float), cudaMemcpyDeviceToDevice);
					//Tk_minus_1 = Tk;
					cudaMemcpy(Tk_minus_1, Tk, numPoints * inputDim * sizeof(float), cudaMemcpyDeviceToDevice);
				}
					
				if (k == 0) {
					
					m->multiply(Tk, theta[k], numPoints, inputDim, outputDim, output[i]);
				}
				else {
					float* temp_out;
					cudaMalloc((void**)&temp_out, numPoints * outputDim * sizeof(float));
					m->multiply(Tk, theta[k], numPoints, inputDim, outputDim, temp_out);
					m->add(output[i], temp_out, numPoints, outputDim, output[i]);
					cudaFree(temp_out);
				}
			}
			m->linearCombination(output[i], output[i], (1.0f / numFilters), 0, numPoints, outputDim, output[i]);
		}

		return output;
	}

	std::vector<float*> GraphConvolutionLayerGPU::backward(std::vector<float*> incomingGradient, float learningRate) {
		std::vector<float*> outgoingGradient;
		MatrixGPU* m = new MatrixGPU();

		m->getIdentityMatrix(numPoints, Tk_minus_2_back);

		int number_of_samples = incomingGradient.size();
		for (int i = 0; i < number_of_samples; i++) {
			float* current_L = L[i];
			float* current_input = X[i];
			float* current_gradient = incomingGradient[i];

			//Tk_minus_1_back = current_L;
			cudaMemcpy(Tk_minus_1_back, current_L, numPoints * numPoints * sizeof(float), cudaMemcpyDeviceToDevice);

			for (int k = 0; k < numFilters; k++) {
				if (k == 0) {
					//Tk_back = Tk_minus_2_back;
					cudaMemcpy(Tk_back, Tk_minus_2_back, numPoints * numPoints * sizeof(float), cudaMemcpyDeviceToDevice);
				}
				else if (k == 1) {
					//Tk_back = Tk_minus_1_back;
					cudaMemcpy(Tk_back, Tk_minus_1_back, numPoints * numPoints * sizeof(float), cudaMemcpyDeviceToDevice);
				}
				else {
					get_chebeshev_polynomial(Tk_minus_1_back, Tk_minus_2_back, current_L, Tk_back, false);
					//Tk_minus_2_back = Tk_minus_1_back;
					cudaMemcpy(Tk_minus_2_back, Tk_minus_1_back, numPoints * numPoints * sizeof(float), cudaMemcpyDeviceToDevice);
					//Tk_minus_1_back = Tk_back;
					cudaMemcpy(Tk_minus_1_back, Tk_back, numPoints * numPoints * sizeof(float), cudaMemcpyDeviceToDevice);
				}

				// Update theta (weights) --> Local Gradient
				m->multiply(Tk_back, current_input, numPoints, numPoints, inputDim, TX);

				m->transpose(TX, numPoints, inputDim, TXT);
				
				m->multiply(TXT, current_gradient, inputDim, numPoints, outputDim, dtheta);

				m->linearCombination(theta[k], dtheta, (1.0f - Parameters::lamba_reg), (-1.0f *learningRate) / numFilters, inputDim, outputDim, theta[k]);
				// Calculate outgoing gradient
				m->multiply(Tk_back, current_gradient, numPoints, numPoints, outputDim, TG);
				m->transpose(theta[k], inputDim, outputDim, thetaT);
				if (k == 0) {
					m->multiply(TG, thetaT, numPoints, outputDim, inputDim, outgoing_gradient[i]);
				}
				else {
					m->multiply(TG, thetaT, numPoints, outputDim, inputDim, temp);
					m->add(outgoing_gradient[i], temp, numPoints, inputDim, outgoing_gradient[i]);
				}
			}
			m->linearCombination(outgoing_gradient[i], outgoing_gradient[i], (1.0f / numFilters), 0, numPoints, inputDim, outgoing_gradient[i]);
		}


		return outgoing_gradient;
	}
};

