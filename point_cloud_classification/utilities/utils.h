#pragma once
#include <iostream>
#include <vector>
#include "../common.h"
#include <iostream>
#include <random>
#include <cmath>

#ifndef imin
#define imin(a,b) (((a)<(b))?(a):(b))
#endif

#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

namespace Utilities {
	inline void genArray(int n, float *a, int seed = 0) {
		if (seed != 0) {
			srand(seed);
		}

		std::random_device rd{};
		std::mt19937 gen{ rd() };
		std::normal_distribution<> d{ 0,0.05 };
		for (int i = 0; i < n; i++) {
			a[i] = d(gen);

			if (a[i] > 1.0f || a[i] < -1.0f) {
				std::cout << a[i] << std::endl;
				a[i] = imax(imin(a[i], 1.0f), - 1.0f);
			}
		}
	}

	inline void printArray(float* arr, int n) {
		for (int i = 0; i < n; i++) {
			std::cout << arr[i] << " ";
		}
		std::cout << std::endl;
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

	inline void printVectorOfFloatsGPU_nonzero(std::vector<float*> arr, int n) {
		float* temp_cpu = (float*)malloc(n * sizeof(float));
		for (int i = 0; i < arr.size(); i++) {
			cudaMemcpy(temp_cpu, arr[i], n * sizeof(float), cudaMemcpyDeviceToHost);
			for (int j = 0; j < n; j++) {
				if(temp_cpu[j] > 0)
					std::cout << j << " ";
			}
		}
		std::cout << std::endl;
	}
}