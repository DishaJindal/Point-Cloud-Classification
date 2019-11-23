#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "../utilities/matrix.h"
#include "../src/testing_helpers.hpp"
#include "layer.h"
#include "graphConvolutionLayer.h"
#include <fstream>
#include <string>

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {

	class GraphConvolutionLayerCPU : public GraphConvolutionLayer {
		GraphConvolutionLayerCPU() {};
	private:
		float* get_chebeshev_polynomial(float* Tk1, float* Tk2, float* L) {
			MatrixCPU* m = new MatrixCPU();

			float* t = (float*)malloc(numPoints * inputDim * sizeof(float));
			m->multiply(L, Tk1, numPoints, numPoints, inputDim, t);
			
			float* Tk = (float*)malloc(numPoints * inputDim * sizeof(float));
			m->linearCombination(t, Tk2, 2, -1, numPoints, inputDim, Tk);
			return Tk;
		}

		Common::PerformanceTimer& timer()
		{
			static Common::PerformanceTimer timer;
			return timer;
		}
	public:
		GraphConvolutionLayerCPU(int numPoints, int inputDim, int outputDim, int batchDim, int numFilters, bool lastLayer) : GraphConvolutionLayer(numPoints, inputDim, outputDim, batchDim, numFilters, lastLayer) {
			for (int i = 0; i < numFilters; i++) {
				float* temp = (float*)malloc(inputDim * outputDim * sizeof(float));
				Utilities::genArray(inputDim * outputDim, temp);
				theta.push_back(temp);
			}
		}

		std::vector<float*> forward(std::vector<float*> inputArg, bool test) {
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
			for (int i = 0; i < batchDim; i++) {
				float* current_input = inputArg[i];
				float* current_L = inputArg[i + batchDim];
				
				Tk_minus_2 = current_input;
				m->multiply(current_L, current_input, numPoints, numPoints, inputDim, Tk_minus_1);

				//Tk_minus_1 = current_L;

				float* current_output = (float*)malloc(numPoints * outputDim * sizeof(float));
				
				
				for (int k = 0; k < numFilters; k++) {
					std::cout << "k = " << k << " ==> ";

					timer().startCpuTimer();
					if(k == 0){
						Tk = Tk_minus_2;
					}else if(k == 1){
						Tk = Tk_minus_1;
					}else{
						Tk = get_chebeshev_polynomial(Tk_minus_1, Tk_minus_2, current_L);
						Tk_minus_2 = Tk_minus_1;
						Tk_minus_1 = Tk;
					}
					timer().endCpuTimer();
					printElapsedTime(timer().getCpuElapsedTimeForPreviousOperation(), "(Generate Chebeshev Polynomial)");

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
					printElapsedTime(timer().getCpuElapsedTimeForPreviousOperation(), "(Calculating output)");
				}
				

				output.push_back(current_output);
			}
			return output;
		}

		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate) {
			std::vector<float*> output;
			return output;
		}
	};
}
