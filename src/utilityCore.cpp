/**
 * @file      utilityCore.cpp
 * @brief     UTILITYCORE: A collection/kitchen sink of generally useful functions
 * @authors   Yining Karl Li
 * @date      2012
 * @copyright Yining Karl Li
 */

#include <iostream>
#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include "utilityCore.hpp"
#include "../point_cloud_classification/utilities/utils.h"

float utilityCore::clamp(float f, float min, float max) {
    if (f < min) {
        return min;
    } else if (f > max) {
        return max;
    } else {
        return f;
    }
}

bool utilityCore::replaceString(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos) {
        return false;
    }
    str.replace(start_pos, from.length(), to);
    return true;
}

std::string utilityCore::convertIntToString(int number) {
    std::stringstream ss;
    ss << number;
    return ss.str();
}

glm::vec3 utilityCore::clampRGB(glm::vec3 color) {
    if (color[0] < 0) {
        color[0] = 0;
    } else if (color[0] > 255) {
        color[0] = 255;
    }
    if (color[1] < 0) {
        color[1] = 0;
    } else if (color[1] > 255) {
        color[1] = 255;
    }
    if (color[2] < 0) {
        color[2] = 0;
    } else if (color[2] > 255) {
        color[2] = 255;
    }
    return color;
}

bool utilityCore::epsilonCheck(float a, float b) {
    if (fabs(fabs(a) - fabs(b)) < EPSILON) {
        return true;
    } else {
        return false;
    }
}

void utilityCore::printCudaMat4(const cudaMat4 &m) {
    utilityCore::printVec4(m.x);
    utilityCore::printVec4(m.y);
    utilityCore::printVec4(m.z);
    utilityCore::printVec4(m.w);
}

glm::mat4 utilityCore::buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
    glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
    glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x, glm::vec3(1, 0, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y, glm::vec3(0, 1, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z, glm::vec3(0, 0, 1));
    glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
    return translationMat * rotationMat * scaleMat;
}

cudaMat4 utilityCore::glmMat4ToCudaMat4(const glm::mat4 &a) {
    cudaMat4 m;
    glm::mat4 aTr = glm::transpose(a);
    m.x = aTr[0];
    m.y = aTr[1];
    m.z = aTr[2];
    m.w = aTr[3];
    return m;
}

glm::mat4 utilityCore::cudaMat4ToGlmMat4(const cudaMat4 &a) {
    glm::mat4 m;
    m[0] = a.x;
    m[1] = a.y;
    m[2] = a.z;
    m[3] = a.w;
    return glm::transpose(m);
}

std::vector<std::string> utilityCore::tokenizeString(std::string str) {
    std::stringstream strstr(str);
    std::istream_iterator<std::string> it(strstr);
    std::istream_iterator<std::string> end;
    std::vector<std::string> results(it, end);
    return results;
}

std::istream& utilityCore::safeGetline(std::istream& is, std::string& t) {
    t.clear();

    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.

    std::istream::sentry se(is, true);
    std::streambuf* sb = is.rdbuf();

    for (;;) {
        int c = sb->sbumpc();
        switch (c) {
        case '\n':
            return is;
        case '\r':
            if (sb->sgetc() == '\n') {
                sb->sbumpc();
            }
            return is;
        case EOF:
            // Also handle the case when the last line has no line ending
            if (t.empty()) {
                is.setstate(std::ios::eofbit);
            }
            return is;
        default:
            t += (char)c;
        }
    }

    return is;
}
//-----------------------------
//-------GLM Printers----------
//-----------------------------

void utilityCore::printMat4(const glm::mat4 &m) {
    std::cout << m[0][0] << " " << m[1][0] << " " << m[2][0] << " " << m[3][0] << std::endl;
    std::cout << m[0][1] << " " << m[1][1] << " " << m[2][1] << " " << m[3][1] << std::endl;
    std::cout << m[0][2] << " " << m[1][2] << " " << m[2][2] << " " << m[3][2] << std::endl;
    std::cout << m[0][3] << " " << m[1][3] << " " << m[2][3] << " " << m[3][3] << std::endl;
}

void utilityCore::printVec4(const glm::vec4 &m) {
    std::cout << m[0] << " " << m[1] << " " << m[2] << " " << m[3] << std::endl;
}

void utilityCore::printVec3(const glm::vec3 &m) {
    std::cout << m[0] << " " << m[1] << " " << m[2] << std::endl;
}



std::vector<glm::vec3> utilityCore::readPointCloud(std::string filename) {
	glm::mat4 centerTransform;
	glm::vec3 t(0.0f, 0.0f, 0.0f);
	glm::vec3 r(3.0f, 0.0f, 1.0f);
	glm::vec3 s(0.01f, 0.001f, 0.001f);
	centerTransform = utilityCore::buildTransformationMatrix(t, r, s);


	std::ifstream fp_in;
	std::vector<glm::vec3> points;
	char* fname = (char*)filename.c_str();
	fp_in.open(fname);
	if (!fp_in.is_open()) {
		throw;
	}
	std::string line;
	utilityCore::safeGetline(fp_in, line);
	if (!line.empty()) {
		replace(line.begin(), line.end(), ',', ' ');
		std::vector<std::string> tokens = tokenizeString(line);
		if (tokens[0] != "OFF") {
			throw;
		}
	}
	utilityCore::safeGetline(fp_in, line);
	int numOfPoints;
	if (!line.empty()) {
		replace(line.begin(), line.end(), ',', ' ');
		std::vector<std::string> tokens = tokenizeString(line);
		numOfPoints = std::stoi(tokens[0]);
		if (numOfPoints < 0) {
			throw;
		}
	}
	while (fp_in.good() && numOfPoints > 0) {
		line;
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty()) {
			replace(line.begin(), line.end(), ',', ' ');
			std::vector<std::string> tokens = tokenizeString(line);
			glm::vec3 pt(atof(tokens[0].c_str()), atof(tokens[1].c_str()), atof(tokens[2].c_str()));
			pt = glm::vec3(centerTransform * glm::vec4(pt, 1));
			points.push_back(pt);
			numOfPoints--;
		}
	}
	return points;
}

std::vector<glm::vec3> utilityCore::farthestSample(std::vector<glm::vec3> &points, int numOfSamplePoints) {
	std::vector<glm::vec3> sampledPoints;
	sampledPoints.push_back(points[0]);
	for (int i = 1; i < numOfSamplePoints; i++) {
		glm::vec3 farthestPoint;
		double maxDistance = 0;
		int index = 0;
		bool foundPoint = false;
		for (int j = 0; j < points.size(); j++) {
			double minDistance = INT_MAX;
			for (int k = 0; k < sampledPoints.size(); k++) {
				double dist = glm::distance(glm::dvec3(sampledPoints[k]), glm::dvec3(points[j]));
				if (dist < minDistance) {
					minDistance = dist;
				}
				if (minDistance < maxDistance) {
					break;
				}
			}
			if (maxDistance < minDistance) {
				maxDistance = minDistance;
				farthestPoint = points[j];
				index = j;
				foundPoint = true;
			}
		}
		if (maxDistance == 0) break;
		points.erase(points.begin() + index);
		sampledPoints.push_back(farthestPoint);
	}
	return sampledPoints;
}

void utilityCore::convertFromVectorToFloatPtr(std::vector<glm::vec3> &points, float *convertedPoints) {
	//float *convertedPoints;
	int numPoints = points.size();
	//convertedPoints = new float[numPoints * 3];
	for (int i = 0; i < numPoints; i++) {
		for (int j = 0; j < 3; j++) {
			convertedPoints[i * 3 + j] = points[i][j];
		}
	}
	//return convertedPoints;
}


std::vector<std::string> utilityCore::get_filenames(std::experimental::filesystem::path path)
{
	namespace stdfs = std::experimental::filesystem;

	std::vector<std::string> filenames;
	using recursive_directory_iterator = std::experimental::filesystem::recursive_directory_iterator;
	for (const auto& dirEntry : recursive_directory_iterator("../data_set/ModelNet10/"))
		filenames.push_back(dirEntry.path().string());
	return filenames;
}



void utilityCore::load_data(std::string folderName, std::vector<float*> &X, std::vector<float*> &Y, std::string subFolder, int numToRead) {
	int count = 0;
	int main_counter = 0;
	std::string previousLabel = "begin";
	for (const auto& name : get_filenames(folderName)) {
		
		int len = name.length();
		if (name[len - 1] == 'f') {
			std::string altName = name;
			replace(altName.begin(), altName.end(), '\\', ' ');
			std::vector<std::string> tokens = tokenizeString(altName);
			std::string subFolderName = tokens[tokens.size() - 2];
			std::string label = tokens[tokens.size() - 3];
			if (label != previousLabel) {
				previousLabel = label;
				main_counter = 0;
			}
			if ((subFolderName == subFolder || subFolder == "all") && (main_counter < numToRead || numToRead == -1)) {
				std::cout << "Reading " << name << std::endl;
				std::vector<glm::vec3> points = readPointCloud(name);
				convertFromVectorToFloatPtr(points, X[count]);
				if (label == "chair") { // 678
					Y[count][2] = 1.0f;
					//float lab[10] = {0,0,1,0,0,0,0,0,0,0};
					//Y.push_back(lab);
				}
				else {

					if (label == "sofa") { // 453
						Y[count][7] = 1.0f;
						//float lab[10] = { 0,0,0,0,0,0,0,1,0,0 };
						//Y.push_back(lab);
					}
					else {

						if (label == "bed") { // 327
							Y[count][1] = 1.0f;
							//float lab[10] = { 0,1,0,0,0,0,0,0,0,0 };
							//Y.push_back(lab);
						}
						else {

							if (label == "toilet") { // 321
								Y[count][9] = 1.0f;
								//float lab[10] = { 0,0,0,0,0,0,0,0,0,1 }; 
								//Y.push_back(lab);
							}
							else {

								if (label == "monitor") { // 191
									Y[count][5] = 1.0f;
									//float lab[10] = { 0,0,0,0,0,1,0,0,0,0 };
									//Y.push_back(lab);
								}
								else {

									if (label == "table") { // 165
										Y[count][8] = 1.0f;
										//float lab[10] = { 0,0,0,0,0,0,0,0,1,0 };
										//Y.push_back(lab);
									}
									else {

										if (label == "dresser") { // 115
											Y[count][4] = 1.0f;
											//float lab[10] = { 0,0,0,0,1,0,0,0,0,0 };
											//Y.push_back(lab);
										}
										else {
											if (label == "desk") { // 109
												Y[count][3] = 1.0f;
												//float lab[10] = { 0,0,0,1,0,0,0,0,0,0 };
												//Y.push_back(lab);
											}
											else {

												if (label == "bathtub") { // 87
													Y[count][0] = 1.0f;
													//float lab[10] = { 1,0,0,0,0,0,0,0,0,0 };
													//Y.push_back(lab);
												}
												else {

													if (label == "night_stand") { //84
														Y[count][6] = 1.0f;
														//float lab[10] = { 0,0,0,0,0,0,1,0,0,0 };
														//Y.push_back(lab);
													}
													//else {
													//	float lab[10] = { 0,0,0,0,0,0,0,0,0,0 };
													//	Y.push_back(lab);
													//}
												}
											}
										}
									}
								}
							}
						}
					}
				}
				main_counter++;
				count++;
			}
		}
	}
}

void utilityCore::normalize_data(float* X, int n) {
	// Mean
	float mean[3] = { 0 };
	for (int i = 0; i < n; i++) {
		mean[0] += X[i * 3];
		mean[1] += X[i * 3 + 1];
		mean[2] += X[i * 3 + 2];
	}
	mean[0] = mean[0] / n;
	mean[1] = mean[1] / n;
	mean[2] = mean[2] / n;
	// Mean center
	for (int i = 0; i < n; i++) {
		X[i * 3] -= mean[0];
		X[i * 3 + 1] -= mean[1];
		X[i * 3 + 2] -= mean[2];
	}
	// Furthest distance
	float furthest_dist = 0;
	for (int i = 0; i < n; i++) {
		float dist = 0;
		dist = (X[i * 3] * X[i * 3]) + (X[i * 3 + 1] * X[i * 3 + 1]) + (X[i * 3 + 2] * X[i * 3 + 2]);
		dist = std::sqrt(dist);
		if (dist > furthest_dist) {
			furthest_dist = dist;
		}
	}
	// Divide by furthest distance
	for (int i = 0; i < n; i++) {
		X[i * 3] /= furthest_dist;
		X[i * 3 + 1] /= furthest_dist;
		X[i * 3 + 2] /= furthest_dist;
	}
}