#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
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

	class GraphConvolutionLayerGPU : public GraphConvolutionLayer {
		GraphConvolutionLayerGPU() {};
	private:
		float* get_chebeshev_polynomial(float* Tk1, float* Tk2, float* L, float mul = false) {
			MatrixGPU* m = new MatrixGPU();

			float* Tk;
			if (mul) {
				float* t;
				cudaMalloc((void**)&t, numPoints * inputDim * sizeof(float));
				m->multiply(L, Tk1, numPoints, numPoints, inputDim, t);

				cudaMalloc((void**)&Tk, numPoints * inputDim * sizeof(float));
				m->linearCombination(t, Tk2, 2, -1, numPoints, inputDim, Tk);
			}
			else {
				float* t;
				cudaMalloc((void**)&t, numPoints * numPoints * sizeof(float));
				m->multiply(L, Tk1, numPoints, numPoints, numPoints, t);

				cudaMalloc((void**)&Tk, numPoints * numPoints * sizeof(float));
				m->linearCombination(t, Tk2, 2, -1, numPoints, numPoints, Tk);
			}
			return Tk;
		}

	public:
		GraphConvolutionLayerGPU(int numPoints, int inputDim, int outputDim, int batchDim, int numFilters, bool lastLayer) : GraphConvolutionLayer(numPoints, inputDim, outputDim, batchDim, numFilters, lastLayer) {
			for (int i = 0; i < numFilters; i++) {
				float* temp;
				cudaMalloc((void**)&temp, inputDim * outputDim * sizeof(float));
				float* temp_cpu = (float*)malloc(inputDim * outputDim * sizeof(float));
				Utilities::genArray(inputDim * outputDim, temp_cpu);
				cudaMemcpy(temp, temp_cpu, inputDim * outputDim * sizeof(float), cudaMemcpyHostToDevice);
				theta.push_back(temp);
			}
		}


		std::vector<float*> forward(std::vector<float*> inputArg, bool test) {
			std::vector<float*> output;
			MatrixGPU* m = new MatrixGPU();

			float* Tk_minus_2;
			cudaMalloc((void**)&Tk_minus_2, numPoints * inputDim * sizeof(float));

			float* Tk_minus_1;
			cudaMalloc((void**)&Tk_minus_1, numPoints * inputDim * sizeof(float));

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

				float* current_output;
				cudaMalloc((void**)&current_output, numPoints * outputDim * sizeof(float));

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
						m->multiply(Tk, theta[k], numPoints, inputDim, outputDim, current_output);
					}
					else {
						float* temp_out;
						cudaMalloc((void**)&temp_out, numPoints * outputDim * sizeof(float));
						m->multiply(Tk, theta[k], numPoints, inputDim, outputDim, temp_out);
						m->add(current_output, temp_out, numPoints, outputDim, current_output);
						//cudaFree(temp_out);
					}
					
				}

				m->linearCombination(current_output, current_output, (1.0f / numFilters), 0, numPoints, outputDim, current_output);
				output.push_back(current_output);
			}
			//cudaFree(Tk_minus_1);
			//cudaFree(Tk_minus_2);
			return output;
		}

		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate) {
			std::vector<float*> outgoingGradient;
			MatrixGPU* m = new MatrixGPU();

			float* TX;
			cudaMalloc((void**)&TX, numPoints * inputDim * sizeof(float));

			float* TXT;
			cudaMalloc((void**)&TXT, numPoints * inputDim * sizeof(float));

			float* dtheta;
			cudaMalloc((void**)&dtheta, inputDim * outputDim * sizeof(float));

			float* Tk_minus_2;
			cudaMalloc((void**)&Tk_minus_2, numPoints * numPoints * sizeof(float));
			m->getIdentityMatrix(numPoints, Tk_minus_2);

			float* Tk_minus_1;
			cudaMalloc((void**)&Tk_minus_1, numPoints * numPoints * sizeof(float));

			float* Tk;

			float* TG;
			cudaMalloc((void**)&TG, numPoints * outputDim * sizeof(float));

			float* temp;
			cudaMalloc((void**)&temp, numPoints * inputDim * sizeof(float));

			float* thetaT;
			cudaMalloc((void**)&thetaT, outputDim * inputDim * sizeof(float));


			int number_of_samples = incomingGradient.size();
			for (int i = 0; i < number_of_samples; i++) {
				float* current_L = L[i];
				float* current_input = X[i];
				float* current_gradient = incomingGradient[i];

				Tk_minus_1 = current_L;

				float* current_outgoing_gradient;
				cudaMalloc((void**)&current_outgoing_gradient, numPoints * inputDim * sizeof(float));

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

					m->linearCombination(theta[k], dtheta, 1, (-1.0f *learningRate) / numFilters, inputDim, outputDim, theta[k]);

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
				
				m->linearCombination(current_outgoing_gradient, current_outgoing_gradient, (1.0f / numFilters), 0, numPoints, inputDim, current_outgoing_gradient);
				outgoingGradient.push_back(current_outgoing_gradient);
			}
			return outgoingGradient;
		}
	};
}
