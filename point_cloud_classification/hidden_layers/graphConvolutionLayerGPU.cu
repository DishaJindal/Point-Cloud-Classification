#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/matrix.h"
#include "layer.h"
#include "graphConvolutionLayer.h"
#include <fstream>
#include <string>
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	float* GraphConvolutionLayerGPU::get_chebeshev_polynomial(float* Tk1, float* Tk2, float* L, float mul) {
		MatrixGPU* m = new MatrixGPU();

		float* Tk;
		if (mul) {
			float* t;
			cudaMalloc((void**)&t, numPoints * inputDim * sizeof(float));
			m->multiply(L, Tk1, numPoints, numPoints, inputDim, t);

			cudaMalloc((void**)&Tk, numPoints * inputDim * sizeof(float));
			m->linearCombination(t, Tk2, 2, -1, numPoints, inputDim, Tk);
			cudaFree(t);
		}
		else {
			float* t;
			cudaMalloc((void**)&t, numPoints * numPoints * sizeof(float));
			m->multiply(L, Tk1, numPoints, numPoints, numPoints, t);

			cudaMalloc((void**)&Tk, numPoints * numPoints * sizeof(float));
			m->linearCombination(t, Tk2, 2, -1, numPoints, numPoints, Tk);
			cudaFree(t);
		}
		return Tk;
	}

	std::vector<float*> GraphConvolutionLayerGPU::forward(std::vector<float*> inputArg, bool test) {
		//std::vector<float*> output;
		MatrixGPU* m = new MatrixGPU();
		float* Tk;

		this->X = std::vector < float* >(inputArg.begin(), inputArg.begin() + batchDim);
		this->L = std::vector < float* >(inputArg.begin() + batchDim, inputArg.end());

		for (int i = 0; i < batchDim; i++) {
			float* current_input = inputArg[i];
			float* current_L = inputArg[i + batchDim];

			// Store data required for backward pass
			//X.push_back(current_input);
			//L.push_back(current_L);

			//cudaMemcpy(Tk_minus_2, current_input, numPoints * inputDim * sizeof(float), cudaMemcpyDeviceToDevice);
			Tk_minus_2 = current_input;
				
			m->multiply(current_L, current_input, numPoints, numPoints, inputDim, Tk_minus_1);

			//Tk_minus_1 = current_L;

			

			//std::cout << "Printing GCN Weights: " << std::endl;
			//Utilities::printVectorOfFloats(theta, 50);
			for (int k = 0; k < numFilters; k++) {
				//std::cout << "k = " << k << " ==> ";

				if (k == 0) {
					Tk = Tk_minus_2;
				}
				else if (k == 1) {
					Tk = Tk_minus_1;
				}
				else {
					Tk = get_chebeshev_polynomial(Tk_minus_1, Tk_minus_2, current_L, true);
					Tk_minus_2 = Tk_minus_1;
					Tk_minus_1 = Tk;
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
			//output.push_back(current_output);
		}
		return output;
	}

	std::vector<float*> GraphConvolutionLayerGPU::backward(std::vector<float*> incomingGradient, float learningRate) {
		std::vector<float*> outgoingGradient;
		MatrixGPU* m = new MatrixGPU();

		//checkCUDAError("cudaMalloc((void**)&Tk_minus_2");
		m->getIdentityMatrix(numPoints, Tk_minus_2_back);

		float* Tk;

		int number_of_samples = incomingGradient.size();
		for (int i = 0; i < number_of_samples; i++) {
			float* current_L = L[i];
			float* current_input = X[i];
			float* current_gradient = incomingGradient[i];

			Tk_minus_1_back = current_L;

			//float* current_outgoing_gradient;
			//cudaMalloc((void**)&current_outgoing_gradient, numPoints * inputDim * sizeof(float));

			for (int k = 0; k < numFilters; k++) {
				if (k == 0) {
					Tk = Tk_minus_2_back;
				}
				else if (k == 1) {
					Tk = Tk_minus_1_back;
				}
				else {
					Tk = get_chebeshev_polynomial(Tk_minus_1_back, Tk_minus_2_back, current_L, false);
					Tk_minus_2_back = Tk_minus_1_back;
					Tk_minus_1_back = Tk;
				}

				// Update theta (weights) --> Local Gradient
				m->multiply(Tk, current_input, numPoints, numPoints, inputDim, TX);

				m->transpose(TX, numPoints, inputDim, TXT);

				m->multiply(TXT, current_gradient, inputDim, numPoints, outputDim, dtheta);

				m->linearCombination(theta[k], dtheta, 1, (-1.0f *learningRate) / numFilters, inputDim, outputDim, theta[k]);

				// Calculate outgoing gradient
				m->multiply(Tk, current_gradient, numPoints, numPoints, outputDim, TG);
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
			//cudaFree(current_outgoing_gradient);
			//outgoingGradient.push_back(current_outgoing_gradient);
		}

		/*cudaFree(TX);
		cudaFree(TXT);
		cudaFree(dtheta);
		cudaFree(Tk_minus_2);
		cudaFree(Tk_minus_1);*/
		//cudaFree(Tk);
		/*cudaFree(TG);
		cudaFree(temp);
		cudaFree(thetaT);*/

		return outgoing_gradient;
	}
};

