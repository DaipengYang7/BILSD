#pragma once
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "utils.h"

#define TEST_EDGESEGMENT 0

using namespace std;

struct EdgeSeg
{
public:
    EdgeSeg(){}
    EdgeSeg(vector<cv::Point> es)
    {
        this->points = es;
    }
    
    // getter
    vector<cv::Point> getPoints()
    {
        return this->points;
    }
    int getLength()
    {
        return this->points.size();
    }
    cv::Point getFront()
    {
        return this->points.front();
    }
    cv::Point getBack()
    {
        return this->points.back();
    }


    void extend(cv::Point p)
    {
        this->points.push_back(p);
    }
    void extend(const EdgeSeg &es)
    {
        this->points.insert(this->points.end(), es.points.begin(), es.points.end());
    }
    cv::Point remove()
    {
        cv::Point p;
        if(this->getLength() >= 1)
        {
            p = this->points.back();
            this->points.pop_back();
        }
        else
        {
            p = cv::Point(0, 0);
        }
        return p;
    }
    void integrateRightPart(EdgeSeg &es)
    {
        // Reverse the points of the left side
        if(this->points.size() > 0)
            std::reverse(this->points.begin(), this->points.end());
        // Extend the points of the right side
        if(es.points.size() > 0)
            this->extend(es);
        // Discard the same points
        if(this->points.size() > 0)
        {
            // 重复点的特点在于，它们是相邻的
            // Note that if there are same points, they are adjacent
            vector<cv::Point> t;
            t.push_back(*(this->points.begin()));
            for(auto p: this->points)
            {
                if((t.end() - 1)->x != p.x || (t.end() - 1)->y != p.y)                          t.push_back(p);
            }
            // Update
            this->points.clear();
            this->points = t;
        }
    }

    vector<cv::Point> points;
};

EdgeSeg dda(int h1, int w1, int h2, int w2);
EdgeSeg dda(double h1, double w1, double h2, double w2);
EdgeSeg dda(cv::Point p1, cv::Point p2);
EdgeSeg dda(cv::Point2d p1, cv::Point2d p2);
