#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "../utilities/kernels.h"
#include "../utilities/utils.h"
#include "layer.h"
#include "globalPoolingLayer.h"
#include <fstream>
#include <string>
#include <cuda.h>
#include "utilities/matrix.h"
#include "device_launch_parameters.h"

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {
 
	class GlobalPoolingLayerGPU : public GlobalPoolingLayer {
		MatrixGPU* m;
		GlobalPoolingLayerGPU() {};
		
	public:
		GlobalPoolingLayerGPU(int inputDim, int outputDim, int batchDim, bool lastLayer) : GlobalPoolingLayer(inputDim, outputDim, batchDim, lastLayer){
			// Allocate space to save mean required for back propagataion
			for (int i = 0; i < batchDim; i++) {
				float* mean_b;
				cudaMalloc((void**)&mean_b, inputDim * sizeof(float));
				this->mean.push_back(mean_b);
			}
			// Allocate space to save argmax required for back propagataion
			for (int i = 0; i < batchDim; i++) {
				int* argmax_b;
				cudaMalloc((void**)&argmax_b, inputDim * sizeof(int));
				this->argMax.push_back(argmax_b);
			}
			this->m = new MatrixGPU();
		}


		void forward(float *inputArg, float *outputArg, bool test) {
			
		}

		void backward(float *incomingGradient, float *outgoingGradient, float learningRate) {

			
		}
	};
}
