#pragma once
#include <string>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "utils.h"
#include "EdgeDetectorHyperparams.h"

#define TEST_EDGEDETECTOR 1

using namespace std;


class EdgeDetector
{
public:
    EdgeDetector(){}

    // getter
    cv::Mat getV1Response();
    cv::Mat getV2ThinnedResponse();
    // For estimated optimal orientation map
    cv::Mat getV2EstiOrient();
    cv::Mat getColoredV2EstiOrient();
    cv::Mat getColoredV2EstiOrientGrid();

//private:
    EdgeDetectorHyperparams hparams;
    string imageName;
    cv::Mat V1Response;
    cv::Mat V2Response;
    cv::Mat V2EstiOrient;
    cv::Mat V2ThinnedResponse;

    void loadImage(string fileName, const cv::Mat &src);
    void V1(const vector<cv::Mat> &pCh);
    void V2();
};

