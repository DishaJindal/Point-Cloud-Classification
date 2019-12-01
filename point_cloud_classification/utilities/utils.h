#pragma once
#include <iostream>
#include <vector>
#include "../common.h"

namespace Utilities {
	inline void genArray(int n, float *a) {
		for (int i = 0; i < n; i++) {
			a[i] = ((2 * ((rand() * 1.0) / RAND_MAX)) - 1) * 0.2;
		}
	}

	inline void printArray(float* arr, int n) {
		for (int i = 0; i < n; i++) {
			std::cout << arr[i];
		}
	}

	inline void printVector(std::vector<float> arr, int n) {
		for (int i = 0; i < n; i++) {
			std::cout << arr[i] << " ";
		}
		std::cout << std::endl;
	}

	inline void printVectorOfFloats(std::vector<float*> arr, int n) {
		for (int i = 0; i < arr.size(); i++) {
			for (int j = 0; j < n; j++) {
				std::cout << arr[i][j] << " ";
			}
			std::cout << std::endl;
		}
	}
}