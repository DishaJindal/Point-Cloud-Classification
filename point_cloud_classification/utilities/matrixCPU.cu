#include "matrix.h"

class MatrixCPU : public Matrix {
	
public:
	MatrixCPU() {};

	/*
		A -> m x n
		B -> n x p
		output = A x B
	*/
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

	/*
		A -> m x n
		B -> p x n
		output = A x B.T (Multiplies A with the transpose of B)
	*/
	void multiplyTranspose(float* A, float* B, int m, int n, int p, float* output){
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < p; j++) {
				float current = 0;
				for (int k = 0; k < n; k++) {
					current += A[i * n + k] * B[j * n + k];
				}
				output[i * p + j] = current;
			}
		}
	}

	/*
		A -> m x n
		output = A.T 
	*/
	void transpose(float* A, int m, int n, float* output){
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				output[i * n + j] = A[j * m + i];
			}
		}
	}


	/*
		A -> m x n
		B -> m x n
		output = A + B
	*/
	void add(float* A, float* B, int m, int n, float* output) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				output[i * n + j] = A[i * n + j] + B[i * n + j];
			}
		}
	}

	/*
		A -> m x n
		B -> m x n
		output = A - B
	*/
	void subtract(float* A, float* B, int m, int n, float* output) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				output[i * n + j] = A[i * n + j] - B[i * n + j];
			}
		}
	}

	/*
		A -> m x n
		B -> m x n
		output = A - alpha * B
	*/
	void subtractWithFactor(float* A, float* B, float alpha, int m, int n, float* output) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				output[i * n + j] = A[i * n + j] - (alpha * B[i * n + j]);
			}
		}
	}

	/*
		Prints matrix A
	*/
	void printMatrix(float* A, int m, int n) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				printf("%f ", A[i * n + j]);
			}
			printf("\n");
		}
	}
};