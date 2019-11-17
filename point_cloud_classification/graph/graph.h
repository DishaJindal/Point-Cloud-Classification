#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "device_launch_parameters.h"

namespace Graph {
	class Graph {
		protected:
			int n;
			float *A;
			float *Lnorm;
		public:
			Graph(float *points, int n, int f_in, int k);
			float* get_A();
			float* get_Lnorm();
		private:
			float* find_adj(float *points, int n, int f_in, int k);
			float* normalize_adj(const float* A);
			float* find_laplacian(const float* Anorm);
			float* normalize_laplacian(const float* L, const int n);
	};
}