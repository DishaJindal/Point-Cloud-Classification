#pragma once
#include "parameters.h"

namespace PointCloudClassification {
	namespace Parameters {
		 int num_neighbours = 4;
		 int num_classes = 10;
		 int num_points = 1024;

		// Network Parameters
		 int batch_size = 10;
		 int num_epochs = 6;//260
		 float learning_rate = 0.001;
		 float keep_prob = 0.5;
		 
		 // Architecture 
		 int input_features = 3;
		 int gcn1_out_features = 1000;
		 int gcn2_out_features = 1000;
		 int fc1_out_features = 600;

		 // Pooling Layer Specific
		 int chebyshev1_order = 2;
		 int chebyshev2_order = 2;

		 // Dropout Probability
		 float keep_drop_prob1 = 0.5;
		 float keep_drop_prob2 = 0.5;
		 float keep_drop_prob3 = 0.5;
		 float keep_drop_prob4 = 0.5;
		 float lamba_reg = 0;// 0.001;
	}
}