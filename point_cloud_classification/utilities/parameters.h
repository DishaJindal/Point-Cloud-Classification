#pragma once

namespace PointCloudClassification {
	namespace Parameters {
		extern int num_neighbours;
		extern int num_classes;
		extern int num_points;

		// Network Parameters
		extern int batch_size;
		extern int num_epochs;
		extern float learning_rate;
		extern float keep_prob;

		// Architecture 
		extern int input_features;
		extern int gcn1_out_features;
		extern int gcn2_out_features;
		extern int fc1_out_features;

		// Pooling Layer Specific
		extern int chebyshev1_order;
		extern int chebyshev2_order;

		// Dropout Probability
		extern float keep_drop_prob1;
		extern float keep_drop_prob2;
		extern float keep_drop_prob3;
		extern float keep_drop_prob4;
	}
}