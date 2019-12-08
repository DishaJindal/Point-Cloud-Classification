/**
* @file      main.cpp
* @brief     Example Boids flocking simulation for CIS 565
* @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
* @date      2013-2017
* @copyright University of Pennsylvania
*/

#include "visualization.hpp"
#include <string>
#include <filesystem>
#include <iomanip>

#define MAX_FRAME_COUNT 1000


namespace PointCloudClassification {
	
	
	GLuint positionLocation = 0;   // Match results from glslUtility::createProgram.
	GLuint velocitiesLocation = 1; // Also see attributeLocations below.
	const char *attributeLocations[] = { "Position", "Velocity" };

	GLuint boidVAO = 0;
	GLuint boidVBO_positions = 0;
	GLuint boidVBO_velocities = 0;
	GLuint boidIBO = 0;
	GLuint displayImage;
	GLuint program[2];

	const unsigned int PROG_BOID = 0;

	float fovy = (float)(PI / 4);
	float zNear = 0.10f;
	float zFar = 10.0f;
	// LOOK-1.2: for high DPI displays, you may want to double these settings.
	int highResolutionMultiplicationFactor = 2.5;
	int width = 1280 * highResolutionMultiplicationFactor;
	int height = 720 * highResolutionMultiplicationFactor;
	int pointSize = 1;

	// For camera controls
	bool leftMousePressed = false;
	bool rightMousePressed = false;
	double lastX;
	double lastY;
	float theta = 1.22f;
	float phi = -0.70f;
	float zoom = 4.0f;
	glm::vec3 lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 cameraPosition;

	glm::mat4 projection;

	// ================
	// Configuration
	// ================


	// LOOK-1.2 - change this to adjust particle count in the simulation
	int N_FOR_VIS;
	vector<glm::vec3> originalPoints;


	/**
	* C visualize function.
	*/
	int visualize(float *points, int numPoints, int trueLabelArg, int predictedLabelArg) {
	//int visualize(string fileName, int numPoints, char* trueLabel, char *predictedLabel) {

		string label[] = { "bathtub", "bed", "chair","desk","dresser","monitor","night_stand","sofa","table","toilet","" };

		originalPoints.clear();

		utilityCore::normalize_data(points, numPoints);
		utilityCore::convertFromFloatPtrToVector(points, originalPoints, numPoints);

		N_FOR_VIS = originalPoints.size();

		const char * trueLabel = label[trueLabelArg].c_str();
		const char *  predictedLabel = (predictedLabelArg < 10 && predictedLabelArg >= 0) ? label[predictedLabelArg].c_str() : NULL;

		if (init(trueLabel, predictedLabel)) {
			mainLoop();
			Scan_Matching::endSimulation();
			return 0;
		}
		else {
			return 1;
		}
	}

	//-------------------------------
	//---------RUNTIME STUFF---------
	//-------------------------------

	std::string deviceName;
	GLFWwindow *window;

	/**
	* Initialization of CUDA and GLFW.
	*/
	bool init(const char* trueLabel, const char *predictedLabel) {
		// Set window title to "Student Name: [SM 2.0] GPU Name"
		cudaDeviceProp deviceProp;
		int gpuDevice = 0;
		int device_count = 0;
		cudaGetDeviceCount(&device_count);
		if (gpuDevice > device_count) {
			std::cout
				<< "Error: GPU device number is greater than the number of devices!"
				<< " Perhaps a CUDA-capable GPU is not installed?"
				<< std::endl;
			return false;
		}
		cudaGetDeviceProperties(&deviceProp, gpuDevice);
		int major = deviceProp.major;
		int minor = deviceProp.minor;

		std::ostringstream ss;
		ss << "True Label : " << trueLabel;

		if (predictedLabel != NULL) {
			ss << " | Predicted Label : " << predictedLabel;
		}

		deviceName = ss.str();

		// Window setup stuff
		glfwSetErrorCallback(errorCallback);

		if (!glfwInit()) {
			std::cout
				<< "Error: Could not initialize GLFW!"
				<< " Perhaps OpenGL 3.3 isn't available?"
				<< std::endl;
			return false;
		}

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
		if (!window) {
			glfwTerminate();
			return false;
		}
		glfwMakeContextCurrent(window);
		glfwSetKeyCallback(window, keyCallback);
		glfwSetCursorPosCallback(window, mousePositionCallback);
		glfwSetMouseButtonCallback(window, mouseButtonCallback);

		glewExperimental = GL_TRUE;
		if (glewInit() != GLEW_OK) {
			return false;
		}

		// Initialize drawing state
		initVAO();

		// Default to device ID 0. If you have more than one GPU and want to test a non-default one,
		// change the device ID.
		cudaGLSetGLDevice(0);

		cudaGLRegisterBufferObject(boidVBO_positions);
		cudaGLRegisterBufferObject(boidVBO_velocities);

		// Initialize N-body simulation
		Scan_Matching::initSimulation(N_FOR_VIS, originalPoints);

		updateCamera();

		initShaders(program);

		glEnable(GL_DEPTH_TEST);

		return true;
	}

	void initVAO() {

		std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (N_FOR_VIS)] };
		std::unique_ptr<GLuint[]> bindices{ new GLuint[N_FOR_VIS] };

		glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
		glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

		for (int i = 0; i < N_FOR_VIS; i++) {
			bodies[4 * i + 0] = 0.0f;
			bodies[4 * i + 1] = 0.0f;
			bodies[4 * i + 2] = 0.0f;
			bodies[4 * i + 3] = 1.0f;
			bindices[i] = i;
		}


		glGenVertexArrays(1, &boidVAO); // Attach everything needed to draw a particle to this
		glGenBuffers(1, &boidVBO_positions);
		glGenBuffers(1, &boidVBO_velocities);
		glGenBuffers(1, &boidIBO);

		glBindVertexArray(boidVAO);

		// Bind the positions array to the boidVAO by way of the boidVBO_positions
		glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions); // bind the buffer
		glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

		glEnableVertexAttribArray(positionLocation);
		glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

		// Bind the velocities array to the boidVAO by way of the boidVBO_velocities
		glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
		glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(velocitiesLocation);
		glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

		glBindVertexArray(0);
	}

	void initShaders(GLuint * program) {
		GLint location;

		program[PROG_BOID] = glslUtility::createProgram(
			"shaders/boid.vert.glsl",
			"shaders/boid.geom.glsl",
			"shaders/boid.frag.glsl", attributeLocations, 2);
		glUseProgram(program[PROG_BOID]);

		if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
			glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
		}
		if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1) {
			glUniform3fv(location, 1, &cameraPosition[0]);
		}
	}

	//====================================
	// Main loop
	//====================================
	void runCUDA() {
		// Map OpenGL buffer object for writing from CUDA on a single GPU
		// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
		// use this buffer

		float4 *dptr = NULL;
		float *dptrVertPositions = NULL;
		float *dptrVertVelocities = NULL;

		cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
		cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);

		Scan_Matching::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);

		// unmap buffer object
		cudaGLUnmapBufferObject(boidVBO_positions);
		cudaGLUnmapBufferObject(boidVBO_velocities);
	}

	void mainLoop() {
		double fps = 0;
		double timebase = 0;
		int frame = 0;
		double frame_counter = 0;

		while (++frame_counter < MAX_FRAME_COUNT && !glfwWindowShouldClose(window)) {
			glfwPollEvents();

			frame++;
			double time = glfwGetTime();

			if (time - timebase > 1.0) {
				fps = frame / (time - timebase);
				timebase = time;
				frame = 0;
			}

			runCUDA();

			std::ostringstream ss;
			ss << deviceName;
			glfwSetWindowTitle(window, ss.str().c_str());

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			glUseProgram(program[PROG_BOID]);
			glBindVertexArray(boidVAO);
			glPointSize((GLfloat)pointSize);
			glDrawElements(GL_POINTS, N_FOR_VIS + 1, GL_UNSIGNED_INT, 0);
			glPointSize(1.0f);

			glUseProgram(0);
			glBindVertexArray(0);

			glfwSwapBuffers(window);
		}
		glfwDestroyWindow(window);
		glfwTerminate();
	}


	void errorCallback(int error, const char *description) {
		fprintf(stderr, "error %d: %s\n", error, description);
	}

	void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, GL_TRUE);
		}
	}

	void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
		leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
		rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	}

	void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
		if (leftMousePressed) {
			// compute new camera parameters
			phi += (xpos - lastX) / width;
			theta -= (ypos - lastY) / height;
			theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
			updateCamera();
		}
		else if (rightMousePressed) {
			zoom += (ypos - lastY) / height;
			zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
			updateCamera();
		}

		lastX = xpos;
		lastY = ypos;
	}

	void updateCamera() {
		cameraPosition.x = zoom * sin(phi) * sin(theta);
		cameraPosition.z = zoom * cos(theta);
		cameraPosition.y = zoom * cos(phi) * sin(theta);
		cameraPosition += lookAt;

		projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
		glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
		projection = projection * view;

		GLint location;

		glUseProgram(program[PROG_BOID]);
		if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
			glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
		}
	}

}