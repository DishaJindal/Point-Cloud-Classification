#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
#include <vector>

using namespace std;

namespace Scan_Matching {
    void initSimulation(int N, vector<glm::vec3>& originalPoints);
    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);
    void endSimulation();

}
