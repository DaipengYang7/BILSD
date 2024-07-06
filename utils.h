#pragma once
#include <string>
#include <vector>
#include <cstdlib>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#define TEST_UTILS 0

using namespace std;
namespace fs = std::experimental::filesystem;

void rmAndMkdir(fs::path dir, bool isRemove = false);
float findMaxValOfMat(const cv::Mat &src);
void writeImage(const cv::Mat &image, const string &name, bool maxValTo1 = false, string dirname="auxiliary");

int computeIdxDis(int i1, int i2, int mod);

float computeDirection(double dx, double dy);
float computeDirection(const cv::Point2d &p1, const cv::Point2d &p2);
float diffDirection(float d1, float d2);
float computeOrientation(double x);
float diffOrientation(float o1, float o2);

cv::Mat filter2D_multiK(const cv::Mat &src, const vector<cv::Mat> &kernels);
cv::Mat findArgMaxOfChannels(const cv::Mat &src);
cv::Mat findMaxOfChannels(const cv::Mat &src);

// Auxiliary functions for visulizing the estimated edge orientation 
cv::Mat visulizeOptimalOrientation(const cv::Mat &estiOptimalOrient, const cv::Mat &edgeVal);
cv::Mat visulizeOptimalOrientationGrid(const cv::Mat &estiOptimalOrient, const cv::Mat &edgeVal);
