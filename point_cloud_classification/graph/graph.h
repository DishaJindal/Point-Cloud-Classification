#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"
#include "device_launch_parameters.h"
//#include "nanoflann.hpp"
//#include <CGAL/Search_traits.h>
//#include <CGAL/point_generators_3.h>
//#include <CGAL/Orthogonal_k_neighbor_search.h>
//#include "Point.h"  // defines types Point, Construct_coord_iterator
//#include "Distance.h"

namespace Graph {
	class GraphCPU {
		protected:
			int n;
			float *A;
			float *Lnorm;
		public:
			GraphCPU(float *points, int n, int f_in, int k);
			float* get_A();
			float* get_Lnorm();
		private:
			/* 
			  Exponent of L2 distance between all nodes
			  Sorts by distance
			  Keeps k nearest
			*/
			void fill_adj(float *points, int n, int f_in, int k);
			
			/*
				I − D^(−1/2) * W * (D^(−1/2))
			*/
			float* normalize_adj(const float* A, int n);
			
			/*
				I - Lnorm
			*/
			float* find_laplacian(const float* Anorm, int n);
			
			/* 
				Lnorm = 2L/λmax − I
			*/
			void fill_normalized_laplacian(float* L, const int n);

	};

	class GraphGPU {
	protected:
		int n;
		float *dev_A;
	public:
		GraphGPU(float *points, float* dev_Lap, int n, int f_in, int k);
		float* get_A();
		float* get_Lnorm();
	private:

		/*
		  Exponent of L2 distance between all nodes
		  Sorts by distance
		  Keeps k nearest
		*/
		void fill_adjA(float *dev_points, float *dev_A, int n, int f_in, int k);

		/*
			I − D^(−1/2) * W * (D^(−1/2))
		*/
		void normalize_adjA(float* dev_A, int n);

		/*
			I - Lnorm
		*/
		void find_laplace(float* dev_A, int n);

		/*
			Lnorm = 2L/λmax − I
		*/
		void fill_normalized_laplace(float* dev_A, int n);
	};
}