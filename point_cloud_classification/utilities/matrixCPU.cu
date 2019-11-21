#include "matrix.h"

/*
	A -> m x n
	B -> n x p
	output = A x B
*/
void MatrixCPU::multiply(float* A, float* B, int m, int n, int p, float* output) {
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
void MatrixCPU::multiplyTranspose(float* A, float* B, int m, int n, int p, float* output){
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
void MatrixCPU::transpose(float* A, int m, int n, float* output){
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			output[j * m + i] = A[i * n + j];
		}
	}
}


/*
	A -> m x n
	B -> m x n
	output = A + B
*/
void MatrixCPU::add(float* A, float* B, int m, int n, float* output) {
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
void MatrixCPU::subtract(float* A, float* B, int m, int n, float* output) {
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
void MatrixCPU::subtractWithFactor(float* A, float* B, float alpha, int m, int n, float* output) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			output[i * n + j] = A[i * n + j] - (alpha * B[i * n + j]);
		}
	}
}

/*
	Prints matrix A
*/
void MatrixCPU::printMatrix(float* A, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			printf("%f ", A[i * n + j]);
		}
		printf("\n");
	}
}
