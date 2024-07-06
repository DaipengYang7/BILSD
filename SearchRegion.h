#pragma once
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <opencv2/core.hpp>
using namespace std;

struct PosDiff
{
    PosDiff():xDiff(0), yDiff(0){}
    PosDiff(int xDiff, int yDiff)
    {
        this->xDiff = xDiff;
        this->yDiff = yDiff;
    }
    int xDiff;
    int yDiff;
};

bool cmp_PosDiff(PosDiff pd1, PosDiff pd2);

class SearchRegion
{
public:
    SearchRegion(int length, int width, float direction);
    int length;
    int width;
    float direction;
    vector<PosDiff> pDiff;

private:
    void constructRegion();
    bool _isLeft(float Ax, float Ay, float Bx, float By, float Px, float Py);
    bool _isRight(float Ax, float Ay, float Bx, float By, float Px, float Py);
};

