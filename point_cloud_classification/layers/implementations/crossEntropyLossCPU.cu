#include "common.h"
#include "../../utilities/kernels.h"
#include "../layer.h"
#include "../crossEntropyLoss.h"
#include <fstream>
#include <string>

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

namespace PointCloudClassification {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	void genArray(int n, float *a) {
		srand(11);

		for (int i = 0; i < n; i++) {
			a[i] = ((2 *((rand() * 1.0 )/ RAND_MAX)) - 1) * 0.0002;
		}
	}

	class CrossEntropyLossCPU : public CrossEntropyLoss {
		CrossEntropyLossCPU() {};
		
		float cost(float *prediction, float *trueLabel, int batchDim, int numClasses) {
			
		}

		void dcost(float *prediction, float *trueLabel, float *gradient, int batchDim, int numClasses) {
			MatrixCPU* m = new MatrixCPU();
			m->subtract(trueLabel, prediction, batchDim, numClasses, gradient);
		}
	};
}
