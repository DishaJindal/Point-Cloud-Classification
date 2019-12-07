#pragma once
#include "parameters.h"

namespace PointCloudClassification {
	namespace Parameters {
		 int num_neighbours = 4;
		 int num_classes = 10;
		 int num_points = 1024;

		// Network Parameters
		 int batch_size = 32;
		 int num_epochs = 20;//260
		 float learning_rate = 0.01;
		 float keep_prob = 1.0f;
		 
		 // Architecture 
		 int input_features = 3;
		 int gcn1_out_features = 1000;
		 int gcn2_out_features = 1000;
		 int fc1_out_features = 600;

		 // Pooling Layer Specific
		 int chebyshev1_order = 4;
		 int chebyshev2_order = 3;

		 // Dropout Probability
		 float keep_drop_prob1 = 1.0f;
		 float keep_drop_prob2 = 1.0f;
		 float keep_drop_prob3 = 1.0f;
		 float keep_drop_prob4 = 1.0f;
		 float lamba_reg = 0.0f;
	}
}