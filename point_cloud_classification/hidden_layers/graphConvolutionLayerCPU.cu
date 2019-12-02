#pragma once

#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "../utilities/matrix.h"
#include "layer.h"
#include "graphConvolutionLayer.h"
#include <fstream>
#include <string>

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	float* GraphConvolutionLayerCPU::get_chebeshev_polynomial(float* Tk1, float* Tk2, float* L, float mul) {
		MatrixCPU* m = new MatrixCPU();

		float* Tk;
		if (mul) {
			float* t = (float*)malloc(numPoints * inputDim * sizeof(float));
			m->multiply(L, Tk1, numPoints, numPoints, inputDim, t);

			Tk = (float*)malloc(numPoints * inputDim * sizeof(float));
			m->linearCombination(t, Tk2, 2, -1, numPoints, inputDim, Tk);
		}
		else {
			float* t = (float*)malloc(numPoints * numPoints * sizeof(float));
			m->multiply(L, Tk1, numPoints, numPoints, numPoints, t);

			Tk = (float*)malloc(numPoints * numPoints * sizeof(float));
			m->linearCombination(t, Tk2, 2, -1, numPoints, numPoints, Tk);
		}
		return Tk;
	}

		
	std::vector<float*> GraphConvolutionLayerCPU::forward(std::vector<float*> inputArg, bool test) {
		std::vector<float*> output;
		MatrixCPU* m = new MatrixCPU();

		float* Tk_minus_2 = (float*) malloc(numPoints * inputDim * sizeof(float));
		float* Tk_minus_1 = (float*) malloc(numPoints * inputDim * sizeof(float));
		float* Tk;

	/*	timer().startCpuTimer();
		m->getIdentityMatrix(numPoints, Tk_minus_2);
		timer().endCpuTimer();
		printElapsedTime(timer().getCpuElapsedTimeForPreviousOperation(), "(Generate Identity Matrix)");
	*/

		this->X = std::vector < float* >(inputArg.begin(), inputArg.begin() + batchDim);
		this->L = std::vector < float* >(inputArg.begin() + batchDim, inputArg.end());

		for (int i = 0; i < batchDim; i++) {
			float* current_input = inputArg[i];
			float* current_L = inputArg[i + batchDim];

			// Store data required for backward pass
			//X.push_back(current_input);
			//L.push_back(current_L);
				
			Tk_minus_2 = current_input;
			m->multiply(current_L, current_input, numPoints, numPoints, inputDim, Tk_minus_1);

			//Tk_minus_1 = current_L;

			float* current_output = (float*)malloc(numPoints * outputDim * sizeof(float));
				
			//std::cout << "Printing GCN Weights: " << std::endl;
			//Utilities::printVectorOfFloats(theta, 50);
			for (int k = 0; k < numFilters; k++) {
				//std::cout << "k = " << k << " ==> ";

				timer().startCpuTimer();
				if(k == 0){
					Tk = Tk_minus_2;
				}else if(k == 1){
					Tk = Tk_minus_1;
				}else{
					Tk = get_chebeshev_polynomial(Tk_minus_1, Tk_minus_2, current_L, true);
					Tk_minus_2 = Tk_minus_1;
					Tk_minus_1 = Tk;
				}
				timer().endCpuTimer();
				//printElapsedTime(timer().getCpuElapsedTimeForPreviousOperation(), "(Generate Chebeshev Polynomial)");

				timer().startCpuTimer();
				/*float* temp = (float*)malloc(numPoints * inputDim * sizeof(float));
				m->multiply(Tk, current_input, numPoints, numPoints, inputDim, temp);*/
				if (k == 0) {
					m->multiply(Tk, theta[k], numPoints, inputDim, outputDim, current_output);
				}
				else {
					float* temp_out = (float*)malloc(numPoints * outputDim * sizeof(float));
					m->multiply(Tk, theta[k], numPoints, inputDim, outputDim, temp_out);
					m->add(current_output, temp_out, numPoints, outputDim, current_output);
				}
				timer().endCpuTimer();
				//printElapsedTime(timer().getCpuElapsedTimeForPreviousOperation(), "(Calculating output)");
			}
				
			m->linearCombination(current_output, current_output, (1.0f/numFilters), 0, numPoints, outputDim, current_output);
			output.push_back(current_output);
		}

		return output;
	}

	std::vector<float*> GraphConvolutionLayerCPU::backward(std::vector<float*> incomingGradient, float learningRate) {
		std::vector<float*> outgoingGradient;
		MatrixCPU* m = new MatrixCPU();

		float* TX = (float*)malloc(numPoints * inputDim * sizeof(float));
		float* TXT = (float*)malloc(numPoints * inputDim * sizeof(float));
		float* dtheta = (float*)malloc(inputDim * outputDim * sizeof(float));

		float* Tk_minus_2 = (float*)malloc(numPoints * numPoints * sizeof(float));
		m->getIdentityMatrix(numPoints, Tk_minus_2);
		float* Tk_minus_1 = (float*)malloc(numPoints * numPoints * sizeof(float));
		float* Tk;

		float* TG = (float*)malloc(numPoints * outputDim * sizeof(float));
			
		float* temp = (float*)malloc(numPoints * inputDim * sizeof(float));
		float* thetaT = (float*)malloc(outputDim * inputDim * sizeof(float));
			

		int number_of_samples = incomingGradient.size();
		for (int i = 0; i < number_of_samples; i++) {
			float* current_L = L[i];
			float* current_input = X[i];
			float* current_gradient = incomingGradient[i];

			Tk_minus_1 = current_L;
			float* current_outgoing_gradient = (float*)malloc(numPoints * inputDim * sizeof(float));

			for (int k = 0; k < numFilters; k++) {
				if (k == 0) {
					Tk = Tk_minus_2;
				}
				else if (k == 1) {
					Tk = Tk_minus_1;
				}
				else {
					Tk = get_chebeshev_polynomial(Tk_minus_1, Tk_minus_2, current_L, false);
					Tk_minus_2 = Tk_minus_1;
					Tk_minus_1 = Tk;
				}	

				// Update theta (weights) --> Local Gradient
				m->multiply(Tk, current_input, numPoints, numPoints, inputDim, TX);
					
				m->transpose(TX, numPoints, inputDim, TXT);

				m->multiply(TXT, current_gradient, inputDim, numPoints, outputDim, dtheta);
					
				m->linearCombination(theta[k], dtheta, 1, (-1.0f *learningRate)/numFilters, inputDim, outputDim, theta[k]);

				// Calculate outgoing gradient
				m->multiply(Tk, current_gradient, numPoints, numPoints, outputDim, TG);
				m->transpose(theta[k], inputDim, outputDim, thetaT);
				if (k == 0) {
					m->multiply(TG, thetaT, numPoints, outputDim, inputDim, current_outgoing_gradient);
				}
				else {
					m->multiply(TG, thetaT, numPoints, outputDim, inputDim, temp);
					m->add(current_outgoing_gradient, temp, numPoints, inputDim, current_outgoing_gradient);
				}
			}
			/*std::cout << "GCN OUTGOING GRAD: " << std::endl;
			std::cout << current_outgoing_gradient[(0 * inputDim) + 0] << " " << current_outgoing_gradient[(0 * inputDim) + 1] << " " << current_outgoing_gradient[(0 * inputDim) + 2] << std::endl;
			std::cout << current_outgoing_gradient[(1 * inputDim) + 0] << " " << current_outgoing_gradient[(1 * inputDim) + 1] << " " << current_outgoing_gradient[(1 * inputDim) + 2] << std::endl;
			std::cout << current_outgoing_gradient[(2 * inputDim) + 0] << " " << current_outgoing_gradient[(2 * inputDim) + 1] << " " << current_outgoing_gradient[(2 * inputDim) + 2] << std::endl;
			std::cout << std::endl;*/
			m->linearCombination(current_outgoing_gradient, current_outgoing_gradient, (1.0f / numFilters), 0, numPoints, inputDim, current_outgoing_gradient);
			outgoingGradient.push_back(current_outgoing_gradient);
		}
		return outgoingGradient;
	}
};

