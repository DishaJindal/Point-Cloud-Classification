#pragma once
#include "parameters.h"

namespace PointCloudClassification {
	namespace Parameters {
		 int num_neighbours = 3;
		 int num_classes = 10;
		 int num_points = 1024;

		// Network Parameters
		 int batch_size = 3;
		 int num_epochs = 260;
		 float learning_rate = 12e-4;

		 int l1_features = 3;

		 float keep_prob = 0.5;
	}
}