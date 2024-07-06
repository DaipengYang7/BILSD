#pragma once
#include <string>
#include <vector>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "utils.h"

#define TEST_EDGEDETECTORHYPERPARAMS 0

using namespace std;

cv::Mat sumOfKernelTo1(const cv::Mat kernel);
cv::Mat equalKernelWeigth(const cv::Mat &kernel);
cv::Mat getAvgKernel(int size);
cv::Mat zeroKernelCentre(cv::Mat kernel, float centreRadius);
cv::Mat getKernelWithRadiusConstraint(cv::Mat kernel, int radius, bool isSumTo1 = true);
cv::Mat discardTinyWeight(cv::Mat kernel, float ratio = 0.3);

// Compute the Gaussian gradient kernel
int computeGaussianSize(double sigma);
cv::Mat getGaussianGradientKernel(float sigma, float seta, float theta);
// Compute the ellipse Gaussian kernel
cv::Mat getEllipseGaussianKernel(float sigma_x, float sigma_y, float theta);
// Compute the Gabor kernel
cv::Mat getGaborKernel(cv::Size ksize, float sigma, float theta, float lambda, float gamma, float psi);

// Compute kernels for edge thinning
void getTBsideKernels(int idx, cv::Mat &topK, cv::Mat &bottomK);

// Auxiliary function for visulizing the kernel with positive and negtive values
cv::Mat visuKernel(const cv::Mat &kernel);
// Auxiliary function for visulizing kernels for edge thinning
cv::Mat mergeTBsideKernels(const cv::Mat &topK, const cv::Mat &bottomK);

struct EdgeDetectorHyperparams
{
public:
    EdgeDetectorHyperparams();
    
    int channels = 1; // 1 channel or 3 channels
    bool ifSqrt = true; // For enhancing the contrast of image
    
    // These hyper-parameters are for the Gaussian gradient kernel, which is used to construct orientation-sensitive neurons in V1
    int numOrient = 6; // The number of orientation cells, better than 12
    float sigmaGG = 1.0;
    float seta = 0.5;
    int GGKernelRadius = 2;
    vector<cv::Mat> GaussianGradientKernels;

    // These hyper-parameters are used to suppress noise edges and texture edges
    int avgKernelSize = 11;
    cv::Mat avgInhiKernel;

    // These hyper-parameters are used to construct orientation-sensitive neurons in V2
    int numOrient_estiOri = 12;
    vector<cv::Mat> orientKernels;
    vector<cv::Mat> topSideKernels;
    vector<cv::Mat> bottomSideKernels;
    // These hyper-parameters are for discarding weak response of V2 orientation-sensitive neurons
    float weakEdgeTh = 0.03;
};
