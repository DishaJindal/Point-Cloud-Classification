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
	void multiply(float* A, float* B, float* output);
	void add(float* A, float* B, float* output);
	void subtract(float* A, float* B, float* output);
	void printMatrix(float* A);
}