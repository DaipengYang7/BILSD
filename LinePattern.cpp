#include "LinePattern.h"

LocalLine::LocalLine()
{
    for(int idx = 0; idx < this->numPattern; idx++)
    {
        SearchRegion leftSR = SearchRegion(this->length, this->width, 360 - idx * 180 / this->numPattern);
        for(int i = 0; i < leftSR.pDiff.size(); i++)
        {
            leftSR.pDiff[i].xDiff += this->length;
            leftSR.pDiff[i].yDiff += this->length;
        }
        leftLinkMasks.push_back(leftSR.pDiff);
        SearchRegion rightSR = SearchRegion(this->length, this->width, 180 - idx * 180 / this->numPattern);
        for(int i = 0; i < rightSR.pDiff.size(); i++)
        {
            rightSR.pDiff[i].xDiff += this->length;
            rightSR.pDiff[i].yDiff += this->length;
        }
        rightLinkMasks.push_back(rightSR.pDiff);
    }
    for(int idx = 0; idx  < 360; idx++)
    {
        SearchRegion SR = SearchRegion(9, 3, idx);
        directionSearchRegion.push_back(SR);
    }
#if TEST_LINEPATTERN
    //for(int i = 0; i < this->numPattern; i++)
    //{
    //    string theta = to_string(i * 180 / this->numPattern);
    //    vector<PosDiff> temp;
    //    cout << "LeftLinkMask: " << endl;
    //    temp = leftLinkMasks[i];
    //    cv::Mat leftLMI = cv::Mat::zeros(this->length * 2 + 1, this->length * 2 + 1, CV_32F);
    //    for(int i = 0; i < temp.size(); i++)
    //    {
    //        cout << temp[i].xDiff << " " << temp[i].yDiff << "    ";
    //        leftLMI.at<float>(temp[i].xDiff, temp[i].yDiff) = 1.;
    //    }
    //    cout << endl;
    //    writeImage(leftLMI, "LeftLMI_" + theta, true);

    //    cout << "RightLinkMask: " << endl;
    //    temp = rightLinkMasks[i];
    //    cv::Mat rightLMI = cv::Mat::zeros(this->length * 2 + 1, this->length * 2 + 1, CV_32F);
    //    for(int i = 0; i < temp.size(); i++)
    //    {
    //        cout << temp[i].xDiff << " " << temp[i].yDiff << "    ";
    //        rightLMI.at<float>(temp[i].xDiff, temp[i].yDiff) = 1.;
    //    }
    //    cout << endl;
    //    writeImage(rightLMI, "RightLMI_" + theta, true);
    //}
#endif
}

