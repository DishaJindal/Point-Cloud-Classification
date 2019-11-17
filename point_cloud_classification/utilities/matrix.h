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
	virtual void add(float* A, float* B, int m, int n, float* output) = 0;
	virtual void subtract(float* A, float* B, int m, int n, float* output) = 0;
	virtual void printMatrix(float* A, int m, int n) = 0;
};



