#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort
#include "common.h"
#include "graph.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "utilities/matrix.h"

#define blockSize 128

namespace Graph {
	using std::vector;
	
	GraphGPU::GraphGPU(float *points, int n, int f_in, int k) {
		// Allocate Memory
		float *dev_points;
		cudaMalloc((void**)&dev_A, n * n * sizeof(float));
		cudaMalloc((void**)&dev_points, n * f_in * sizeof(float));
		cudaMemcpy(dev_points, points, n * f_in * sizeof(float), cudaMemcpyHostToDevice);
		
		fill_adjA(dev_points, dev_A, n, f_in, k);
		normalize_adjA(dev_A, n);
		find_laplace(dev_A, n);
		fill_normalized_laplace(dev_A, n);
		
		// Free Memory
		cudaFree(dev_points);
	};


	float* GraphGPU::get_A() {
		return dev_A;
	};

	float* GraphGPU::get_Lnorm() {
		return dev_A;
	};

	__global__ void kernel_find_dist(int n, int f_in, float* points, float* A) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int i = idx / n;
		int j = idx % n;
		A[idx] = 0;
		float* p1 = &points[i * f_in];
		float* p2 = &points[j * f_in];
		for (int k = 0; k < f_in; k++) {
			A[idx] += (p1[k] - p2[k]) * (p1[k] - p2[k]);
		}
		A[idx] = exp(-1 * A[idx]);
	}

	__global__ void kernel_mark_top_k(float *dev_A, int n, int k) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		for (int i = 0; i < k; i++) {
			float k_min = FLT_MAX;
			int k_idx = -1;
			for (int j = 0; j < n; j++) {
				if ((dev_A[n * idx + j] >= 0.0f) & (dev_A[n * idx + j] < k_min)) {
					k_min = dev_A[n * idx + j];
					k_idx = n * idx + j;
				}
			}
			dev_A[k_idx] = dev_A[k_idx] * -1.0f;
		}
	}

	__global__ void kernel_update_dist(float *dev_A) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (dev_A[idx] < 0) {
			dev_A[idx] = dev_A[idx] * -1.0f;
		}
		else {
			dev_A[idx] = 0.0;
		}
	}

	void GraphGPU::fill_adjA(float *dev_points, float *dev_A, int n, int f_in, int k) {
		dim3 nsquareBlocks((n*n + blockSize - 1) / blockSize);
		kernel_find_dist << < nsquareBlocks, blockSize >> > (n, f_in, dev_points, dev_A);

		dim3 nBlocks((n + blockSize - 1) / blockSize);
		kernel_mark_top_k << <nBlocks, blockSize >> > (dev_A, n, k);

		kernel_update_dist << < nsquareBlocks, blockSize >> > (dev_A);
	};
	
	__global__ void kernel_pow(float *scan_temp) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		float raise = -0.5f;
		scan_temp[idx] = pow(scan_temp[idx], raise);
	}

	__global__ void kernel_vec_mat(float *dev_A, float *D, int n) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int i = idx / n;
		int j = idx % n;
		dev_A[n*i + j] = dev_A[n*i + j] * D[i];
	}

	__global__ void kernel_mat_vec(float *dev_A, float *D, int n) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int i = idx / n;
		int j = idx % n;
		dev_A[n*j + i] = dev_A[n*j + i] * D[i];
	}

	void GraphGPU::normalize_adjA(float* dev_A, int n) {
		float* devA_temp;
		float* scan_temp;
		MatrixGPU* m = new MatrixGPU();

		// Mem Allocate
		cudaMalloc((void**)&devA_temp, n * n * sizeof(float));
		cudaMalloc((void**)&scan_temp, n * n * sizeof(float));

		dim3 nBlocks((n + blockSize - 1) / blockSize);
		dim3 nsquareBlocks((n*n + blockSize - 1) / blockSize);

		// Transpose A
		m->transpose(dev_A, n, n, devA_temp);
		// Dist Sum
		m->sumAcrossDim1(devA_temp, n, n, scan_temp);
		// Raise to power -1/2
		kernel_pow << <nBlocks, blockSize >> > (scan_temp);
		// Anorm = D(-1/2) * A
		kernel_vec_mat << <nsquareBlocks, blockSize >> > (dev_A, scan_temp, n);
		// Anorm = Anorm * D(-1/2)
		kernel_mat_vec << <nsquareBlocks, blockSize >> > (dev_A, scan_temp, n);

		// Mem Free
		cudaFree(devA_temp);
		cudaFree(scan_temp);
	};

	__global__ void kernel_laplace(float *dev_A, int n) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int i = idx / n;
		int j = idx % n;
		if (i == j)
			dev_A[n*i + j] = 1 - dev_A[n*i + j];
		else
			dev_A[n*i + j] = 0 - dev_A[n*i + j];
	}

	void GraphGPU::find_laplace(float* dev_A, int n) {
		dim3 nsquareBlocks((n*n + blockSize - 1) / blockSize);
		kernel_laplace << <nsquareBlocks, blockSize >> > (dev_A, n);
	};

	__global__ void kernel_norm_laplace(float *dev_A, int n, float max_eigen) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int i = idx / n;
		int j = idx % n;
		if (i == j)
			dev_A[n*i + j] = ((2 * dev_A[n*i + j]) / max_eigen) - 1;
		else
			dev_A[n*i + j] = (2 * dev_A[n*i + j]) / max_eigen;
	}

	void GraphGPU::fill_normalized_laplace(float* dev_A, int n) {
		dim3 nsquareBlocks((n*n + blockSize - 1) / blockSize);
		float max_eigen = 2;
		kernel_norm_laplace << <nsquareBlocks, blockSize >> > (dev_A, n, max_eigen);
	};
}