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

	class GraphConvolutionLayerCPU : public GraphConvolutionLayer {
		GraphConvolutionLayerCPU() {};
	private:
		float* get_chebeshev_polynomial(float* Tk1, float* Tk2, float* L) {
			float* t = (float*)malloc(numPoints * numPoints * sizeof(float));
			MatrixCPU* m = new MatrixCPU();
			m->multiply(L, Tk1, numPoints, numPoints, numPoints, t);
			
			float* Tk = (float*)malloc(numPoints * numPoints * sizeof(float));
			m->linearCombination(t, Tk2, 2, -1, numPoints, numPoints, Tk);
			return Tk;
		}
	public:
		GraphConvolutionLayerCPU(int numPoints, int inputDim, int outputDim, int batchDim, int numFilters, bool lastLayer) : GraphConvolutionLayer(numPoints, inputDim, outputDim, batchDim, numFilters, lastLayer) {
		
		}

		std::vector<float*> forward(std::vector<float*> inputArg, bool test) {
			std::vector<float*> output;
			MatrixCPU* m = new MatrixCPU();

			float* Tk_minus_2 = (float*) malloc(numPoints * numPoints * sizeof(float));
			float* Tk_minus_1 = (float*) malloc(numPoints * numPoints * sizeof(float));
			float* Tk;

			m->getIdentityMatrix(numPoints, Tk_minus_2);

			for (int i = 0; i < batchDim; i++) {
				float* current_input = inputArg[i];
				float* current_L = inputArg[i + batchDim];
				Tk_minus_1 = current_L;

				float* current_output = (float*)malloc(numPoints * outputDim * sizeof(float));
				
				for (int k = 0; k < numFilters; k++) {
					if(k == 0){
						Tk = Tk_minus_2;
					}else if(k == 1){
						Tk = Tk_minus_1;
					}else{
						Tk = get_chebeshev_polynomial(Tk_minus_1, Tk_minus_2, current_L);
						Tk_minus_2 = Tk_minus_1;
						Tk_minus_1 = Tk;
					}
					float* temp = (float*)malloc(numPoints * inputDim * sizeof(float));
					m->multiply(Tk, current_input, numPoints, numPoints, inputDim, temp);
					if (k == 0) {
						m->multiply(temp, theta[k], numPoints, inputDim, outputDim, current_output);
					}
					else {
						float* temp_out = (float*)malloc(numPoints * outputDim * sizeof(float));
						m->multiply(temp, theta[k], numPoints, inputDim, outputDim, temp_out);
						m->add(current_output, temp_out, numPoints, outputDim, current_output);
					}
				}
				output.push_back(current_output);
			}
			return output;
		}

		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate) {

		}
	};
}
