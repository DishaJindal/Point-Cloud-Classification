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
		float* get_chebeshev_polynomial(float* L, int k) {
			if (k == 0) {
				return nullptr;
			}else if (k == 1) {
				return L;
			}
			else {
				MatrixCPU* m = new MatrixCPU();
				float* Tk1 = get_chebeshev_polynomial(L, k - 1);
				float* Tk2 = get_chebeshev_polynomial(L, k - 2);
				float* t = (float*)malloc(numPoints * numPoints * sizeof(float));
				m->multiply(L, Tk1, numPoints, numPoints, numPoints, t);
				if (Tk2) {
					float* Tk = (float*)malloc(numPoints * numPoints * sizeof(float));
					m->subtract(t, Tk2, numPoints, numPoints, Tk);
					return Tk;
				}
				else {
					//float* Tk = (float*)malloc(numPoints * numPoints * sizeof(float));
					m->subtractIdentity(t, numPoints);
					return t;
				}

			}
		}
	public:
		GraphConvolutionLayerCPU(int numPoints, int inputDim, int outputDim, int batchDim, int numFilters, bool lastLayer) : GraphConvolutionLayer(numPoints, inputDim, outputDim, batchDim, numFilters, lastLayer) {
		
		}

		std::vector<float*> forward(std::vector<float*> inputArg, bool test) {
			std::vector<float*> output;

			MatrixCPU* m = new MatrixCPU();
			for (int i = 0; i < batchDim; i++) {
				float* current_input = inputArg[i];
				float* current_L = inputArg[i + batchDim];
				float* current_output = (float*)malloc(numPoints * outputDim * sizeof(float));
				//memset(current_output, 0, numPoints * outputDim * sizeof(float));
				for (int k = 0; k < numFilters; k++) {
					float* Tk = get_chebeshev_polynomial(current_L, k);
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
