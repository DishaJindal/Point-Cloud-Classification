#pragma once

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <math.h>

class Matrix{
public:
	virtual void multiply(float* A, float* B, int m, int n, int p, float* output) = 0;
	virtual void multiplyTranspose(float* A, float* B, int m, int n, int p, float* output) = 0;
	virtual void transpose(float* A, int m, int n, float* output) = 0;
	virtual void subtractWithFactor(float* A, float* B, float alpha, int m, int n, float* output) = 0;
	virtual void add(float* A, float* B, int m, int n, float* output) = 0;
	virtual void subtract(float* A, float* B, int m, int n, float* output) = 0;
	virtual void printMatrix(float* A, int m, int n) = 0;
	virtual void subtractIdentity(float* A, int m) = 0;
	virtual void getIdentityMatrix(int m, float* A) = 0;
	virtual void linearCombination(float* A, float* B, float alpha, float beta, int m, int n, float* output) = 0;
};

class MatrixCPU : public Matrix {
public:
	void multiply(float* A, float* B, int m, int n, int p, float* output);
	void multiplyTranspose(float* A, float* B, int m, int n, int p, float* output);
	void transpose(float* A, int m, int n, float* output);
	void subtractWithFactor(float* A, float* B, float alpha, int m, int n, float* output);
	void add(float* A, float* B, int m, int n, float* output);
	void subtract(float* A, float* B, int m, int n, float* output);
	void printMatrix(float* A, int m, int n);
	void subtractIdentity(float* A, int m);
	void getIdentityMatrix(int m, float* A);
	void linearCombination(float* A, float* B, float alpha, float beta, int m, int n, float* output);
	void maxAcrossDim1(float* A, int  n, int m, int* argmaxOutput, float* output);
	void meanAcrossDim1(float * A, int m, int n, float * output);
	void varianceAcrossDim1(float * A, int m, int n, float * output, float* mean);
};

class MatrixGPU : public Matrix {
public:
	void multiply(float* A, float* B, int m, int n, int p, float* output);
	void multiplyTranspose(float* A, float* B, int m, int n, int p, float* output);
	void transpose(float* A, int m, int n, float* output);
	void subtractWithFactor(float* A, float* B, float alpha, int m, int n, float* output);
	void add(float* A, float* B, int m, int n, float* output);
	void subtract(float* A, float* B, int m, int n, float* output);
	void printMatrix(float* A, int m, int n);
};



