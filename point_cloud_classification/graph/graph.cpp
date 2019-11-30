#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort
#include "common.h"
#include "graph.h"
#include "device_launch_parameters.h"

#include <Eigen/Dense>
#include <Eigen/Core>

//#include <eigen-eigen-323c052e1731/Eigen>
//#include <eigen3/Eigen/Core>


//#include <eigen3/Eigen/Eigenvalues>
//#include "eigen.h"
//#include <Eigen/Eigenvalues> 

using namespace Eigen;

namespace Graph {
	using std::vector;

	Graph::Graph(float *points, int n, int f_in, int k) {
		fill_adj(points, n, f_in, k);
		float* Anorm = normalize_adj(A, n);
		float* L = find_laplacian(Anorm, n);
		fill_normalized_laplacian(L, n);
	};

	vector<int> sort_indexes(const vector<float> &v) {

		// initialize original index locations
		vector<int> idx(v.size());
		iota(idx.begin(), idx.end(), 0);
		// sort indexes based on comparing values in v
		sort(idx.begin(), idx.end(), [&v](float i1, float i2) {return v[i1] < v[i2]; });
		return idx;
	}

	float* Graph::get_A() {
		return A;
	};

	float* Graph::get_Lnorm() {
		return Lnorm;
	};

	void Graph::fill_adj(float *points, int n, int f_in, int k) {
		A = new float[n * n];
		for (int i = 0; i < n; i++) {

			// Find distance( e^(-d^2)) of point i with all other points
			for (int j = 0; j < n; j++) {
				int idx = n * i + j;
				A[idx] = 0;
				float* p1 = &points[i * f_in];
				float* p2 = &points[j * f_in];
				for (int k = 0; k < f_in; k++) {
					A[idx] += (p1[k] - p2[k]) * (p1[k] - p2[k]);
				}
				A[idx] = exp(-1 * A[idx]);
			}

			// Sort these distances and zero out the ones which are not top k
			vector<float> dist(A + n*i, A + n * (i+1));
			int counter = 0;
			for (auto idx : sort_indexes(dist)) {
				counter++;
				if (counter >= k)
					A[n * i + idx] = 0;
			}
		}
	};
	
	float* Graph::normalize_adj(const float* A, int n) {
		float* Anorm = new float[n * n];
		vector<float> d(n, 0);
		for (int i = 0; i < n; i++) {
			// Sum of k neighbours of each point
			for (int j = 0; j < n; j++) {
				d[i] += A[n*i + j];
			}
			// Raise to power -1/2
			d[i] = std::pow(d[i], -0.5);
		}

		// Anorm = D(-1/2) * A
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				Anorm[n*i + j] = A[n*i + j] * d[i];
			}
		}

		// Anorm = Anorm * D(-1/2)
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				Anorm[n*j + i] = Anorm[n*j + i] * d[i];
			}
		}
		return Anorm;
	};

	float* Graph::find_laplacian(const float* Anorm, int n) {
		float* L = new float[n * n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (i == j)
					L[n*i + j] = 1 - Anorm[n*i + j];
				else
					L[n*i + j] = 0 - Anorm[n*i + j];
			}
		}
		return L;
	};

	void Graph::fill_normalized_laplacian(const float* L, const int n) {
		Lnorm = new float[n * n];

		// Trying to incorporate eigen
		MatrixXf eigenX = Map<MatrixXf>(Lnorm, n, n);
		VectorXcf eivals = eigenX.eigenvalues();

		float max_eigen = 2;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (i == j)
					Lnorm[n*i + j] = ((2 * L[n*i + j]) / max_eigen) - 1;
				else
					Lnorm[n*i + j] = (2 * L[n*i + j]) / max_eigen;
			}
		}
	};
}