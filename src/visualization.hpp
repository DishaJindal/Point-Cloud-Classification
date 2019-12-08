#pragma once

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "utilityCore.hpp"
#include "glslUtility.hpp"
#include "kernel.h"
#include <vector>
using namespace std;

namespace PointCloudClassification {
	//====================================
	// GL Stuff
	//====================================

	extern GLuint positionLocatio;   // Match results from glslUtility::createProgram.
	extern GLuint velocitiesLocation; // Also see attributeLocations below.
	extern const char *attributeLocations[];

	extern GLuint boidVAO;
	extern GLuint boidVBO_positions;
	extern GLuint boidVBO_velocities;
	extern GLuint boidIBO;
	extern GLuint displayImage;
	extern GLuint program[2];

	extern const unsigned int PROG_BOID;

	extern float fovy;
	extern float zNear;
	extern float zFar;
	// LOOK-1.2: for high DPI displays, you may want to double these settings.
	extern int highResolutionMultiplicationFactor;
	extern int width;
	extern int height;
	extern int pointSize;

	// For camera controls
	extern bool leftMousePressed;
	extern bool rightMousePressed;
	extern double lastX;
	extern double lastY;
	extern float theta;
	extern float phi;
	extern float zoom;
	extern glm::vec3 lookAt;
	extern glm::vec3 cameraPosition;

	extern glm::mat4 projection;

	// ================
	// Configuration
	// ================


	// LOOK-1.2 - change this to adjust particle count in the simulation
	extern int N_FOR_VIS;
	extern vector<glm::vec3> originalPoints;
	extern glm::mat4 randomTransform;
	extern glm::mat4 centerTransform;

	//====================================
	// Main
	//====================================

	int visualize(float *points, int numPoints, int trueLabel, int predictedLabel);
	//int visualize(std::string s, int numPoints, char* trueLabel, char *predictedLabel);

	//====================================
	// Main loop
	//====================================
	void mainLoop();
	void errorCallback(int error, const char *description);
	void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
	void updateCamera();
	void runCUDA();

	//====================================
	// Setup/init Stuff
	//====================================
	bool init(const char* trueLabel, const char *predictedLabel);
	void initVAO();
	void initShaders(GLuint *program);


}
