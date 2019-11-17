#include "matrix.h"

class MatrixCPU : public Matrix {
	
public:
	MatrixCPU() {};

	void multiply(float* A, float* B, int m, int n, int p, float* output) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < p; j++) {
				float current = 0;
				for (int k = 0; k < n; k++) {
					current += A[i * n + k] * B[k * p + j];
				}
				output[i * p + j] = current;
			}
		}
	}

	void add(float* A, float* B, int m, int n, float* output) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				output[i * n + j] = A[i * n + j] + B[i * n + j];
			}
		}
	}

	void subtract(float* A, float* B, int m, int n, float* output) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				output[i * n + j] = A[i * n + j] - B[i * n + j];
			}
		}
	}

	void printMatrix(float* A, int m, int n) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				printf("%f ", A[i * n + j]);
			}
			printf("\n");
		}
	}
};