#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "graph.h"
#include "device_launch_parameters.h"

namespace Graph {

	Graph::Graph(float *points, int n, int f_in, int k) {
		A = find_adj(points, n, f_in, k);
		float* Anorm = normalize_adj(A);
		float* L = find_laplacian(Anorm);
		Lnorm = normalize_laplacian(L, n);
	};

	float* Graph::get_A() {
		return A;
	};

	float* Graph::get_Lnorm() {
		return Lnorm;
	};

	float* Graph::find_adj(float *points, int n, int f_in, int k) {
		A = new float[n * n];
		// e^(-d^2) between all points
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				int idx = (n - 1) * i + j;
				A[idx] = 0;
				float* p1 = &points[i * f_in];
				float* p2 = &points[j * f_in];
				for (int k = 0; k < f_in; k++) {
					A[idx] += (p1[k] - p2[k]) * (p1[k] - p2[k]);
				}
				A[idx] = exp(-1 * A[idx]);
			}
		}

		// sort
		// top k
		return A;
	};
	
	float* Graph::normalize_adj(const float* A) {

	};

	float* Graph::find_laplacian(const float* Anorm) {

	};

	float* Graph::normalize_laplacian(const float* L, const int n) {
		Lnorm = new float[n * n];
		
		return Lnorm;
	};
}