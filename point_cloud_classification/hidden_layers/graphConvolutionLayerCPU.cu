#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
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

	public:
		GraphConvolutionLayerCPU(int numPoints, int inputDim, int outputDim, int batchDim, int numFilters, bool lastLayer) : GraphConvolutionLayer(numPoints, inputDim, outputDim, batchDim, numFilters, lastLayer) {
		
		}

		std::vector<float*> forward(std::vector<float*> inputArg, bool test) {

		}

		std::vector<float*> backward(std::vector<float*> incomingGradient, float learningRate) {

		}
	};
}
