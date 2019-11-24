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

		extern int l1_features;
		extern float keep_prob;
	}
}