#pragma once
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "utils.h"
#include "EdgeSegment.h"

#define TEST_LINESEGMENT 0

using namespace std;

struct LineSeg
{
public:

    LineSeg(){}
    LineSeg(const cv::Point &p1, const cv::Point &p2);
    void addFirstTwoPoint(const cv::Point &p1, const cv::Point &p2);

    EdgeSeg edgeSeg;
    // line equation: y = ax + b (Note that cv::Point x->cols y->rows)
    // A = \sum_{x_i^2}
    // B = \sum_{x_i}
    // C = \sum_{x_iy_i}
    // D = \sum_{y_i}
    // E = \sum_{y_i^2} // For isInverseXY = true
    // a = (nC - BD) / (nA - B^2)
    // b = (DA - BC) / (nA - B^2)
    // rx + sy + t = 0 --> r = (nC - BD) | s = (B^2 - nA) | t = (DA - BC)
    // k = (r^2 + s^2)^{1/2} --> r/kx + s/ky + t/k = 0 --> the distance between "Point(x_0, y_0)" and "line" is "|r/kx_0 + s/ky_0 + t/k|"
    double A, B, C, D, E; // Can not use FLOAT type, because FLOAT type only guarantes 7 significant digits while the subsequent multiplication will result in integers more than 7 significant digits
    double lineCo[3]; // r/k, s/k, t/k
    bool isInverseXY = false; // If "slope > 1", swap X-axis and Y-axis, otherwise as the slope increases, the deviation will be bigger
    cv::Point2d startP;
    cv::Point2d endP;
    float orientation; // (0, 180]
    double avgDisTh = 0.5; // threshold: the distance between the edge point and the estimated line
    double curDisTh = 1.0; // threshold: the distance between the current point and the estimated line
    double totalDis; // The total distance between edge points and the estimated line
    double avgDis; // The average distance between edge points and the estimated line
    bool canAddEndP = true;

    void addEndP(cv::Point &p);
    void removeEndP();

    void LSE();
    void getEndpoints();
    float computeAvgDis();
    void update();

    bool isValidLineSeg()
    {
        return this->edgeSeg.getLength();
    }
    double computeLength()
    {
        return sqrt(pow(endP.x - startP.x, 2) + pow(endP.y - startP.y, 2));
    }
    EdgeSeg convertToEdgeSeg()
    {
        return dda(this->startP, this->endP);
    }
};

LineSeg connectTwoLine(LineSeg ls1, LineSeg ls2, cv::Point p1, cv::Point p2);
