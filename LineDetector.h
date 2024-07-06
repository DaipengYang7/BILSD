#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "utils.h"
#include "EdgeSegment.h"
#include "LineSegment.h"
#include "LinePattern.h"
#include "SearchRegion.h"

#define TEST_LINEDETECTOR 1

using namespace std;

struct LDhyperparams
{
public:
    // These hyper-parameters are for connecting gaps
    int skipGapLength = 9; // 7 时，效果下降
    int skipGapWidth = 3; // 5 时，效果差不少
    float diffDirectionTh = 5;
    float diffOrientationTh = 5;
};

class LineDetector
{
public:
    LineDetector(){}

    void connectEdge(string filename, cv::Mat edge, cv::Mat thinnedEdge, cv::Mat estOri);

//private:
    LDhyperparams hparams;
    LocalLine LL;
    string imageName;
    cv::Mat edge;
    cv::Mat thinnedEdge;
    cv::Mat AO;
    vector<EdgeSeg> ES;

    vector<LineSeg> LS;
    vector<LineSeg> LSWoGap;
    vector<LineSeg> disappeardLS;

    EdgeSeg connectOneSide(bool isLeftSide, const cv::Mat &paddingThinnedEdge, const cv::Mat & paddingThinnedEdgeMask, cv::Mat &markMask, int initX, int initY, int initArgmaxO);
    void V4();
    vector<cv::Point> connectTwoPoints(int h1, int w1, int h2, int w2);

    void generateLineSeg();

    void connectGap();

    float computeSalience(const EdgeSeg &es);

    cv::Mat visulizeEdgeSeg(const cv::Mat &src);
    cv::Mat visulizeLineSeg(const cv::Mat &src);
};
