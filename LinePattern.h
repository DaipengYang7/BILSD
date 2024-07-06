#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "utils.h"
#include "SearchRegion.h"

#define TEST_LINEPATTERN 1

using namespace std;

class LocalLine
{
public:
    LocalLine();
    int numPattern = 12;
    int length = 3;
    int width = 3;
    vector<vector<PosDiff>> leftLinkMasks;
    vector<vector<PosDiff>> rightLinkMasks;

    vector<SearchRegion> directionSearchRegion;
};
