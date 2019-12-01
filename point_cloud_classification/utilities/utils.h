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

	inline void printArrayGPU(float* arr, int n) {
		float* temp_cpu = (float*)malloc(n * sizeof(float));
		cudaMemcpy(temp_cpu, arr, n * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < n; i++) {
			std::cout << temp_cpu[i] << " ";
		}
		std::cout << std::endl;
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

	inline void printVectorOfFloatsGPU(std::vector<float*> arr, int n) {
		float* temp_cpu = (float*)malloc(n * sizeof(float));
		for (int i = 0; i < arr.size(); i++) {
			cudaMemcpy(temp_cpu, arr[i], n * sizeof(float), cudaMemcpyDeviceToHost);
			for (int j = 0; j < n; j++) {
				std::cout << temp_cpu[j] << " ";
			}
			std::cout << std::endl;
		}
	}
}