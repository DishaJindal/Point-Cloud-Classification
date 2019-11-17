#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "device_launch_parameters.h"

namespace Graph {
	class Graph {
		protected:
			int n;
			float *A;
			float *Lnorm;
		public:
			Graph(float *points, int n, int f_in, int k);
			float* get_A();
			float* get_Lnorm();
		private:
			/* 
			  Exponent of L2 distance between all nodes
			  Sorts by distance
			  Keeps k nearest
			*/
			float* find_adj(float *points, int n, int f_in, int k);
			
			/*
				I − D^(−1/2) * W * (D^(−1/2))
			*/
			float* normalize_adj(const float* A);
			
			/*
				I - Lnorm
			*/
			float* find_laplacian(const float* Anorm);
			
			/* 
				Lnorm = 2L/λmax − I
			*/
			float* normalize_laplacian(const float* L, const int n);
	};
}