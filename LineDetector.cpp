#include  "LineDetector.h"

void LineDetector::connectEdge(string filename, cv::Mat edge, cv::Mat thinnedEdge, cv::Mat estOri)
{
    this->imageName = filename;
    this->edge = edge;
    this->thinnedEdge = thinnedEdge;
    this->AO = estOri;
    vector<EdgeSeg>().swap(this->ES);
    vector<LineSeg>().swap(this->LS);
    vector<LineSeg>().swap(this->LSWoGap);
    vector<LineSeg>().swap(this->disappeardLS);

    //clock_t start = clock();

    // Connect edge points and generate edge segments
    this->V4();
    //cout << "edge segments generation cost: " << double(clock() - start) / CLOCKS_PER_SEC << " s" << endl;
    writeImage(this->visulizeEdgeSeg(this->thinnedEdge), this->imageName + "_continousEdges", false, "ed_visu");
    //writeImage(visulizeOptimalOrientation(this->AO, this->thinnedEdge), this->imageName + "_seg_AO", false, "ed_visu");
    //writeImage(visulizeOptimalOrientationGrid(this->AO, this->thinnedEdge), this->imageName + "_seg_AOGrid", false, "ed_visu");

    // Generate line segments
    generateLineSeg();
    writeImage(this->visulizeLineSeg(this->thinnedEdge), this->imageName + "_seg_lines", false, "ed_visu");
    //cout << "line segments generation cost: " << double(clock() - start) / CLOCKS_PER_SEC << " s" << endl;

    // Skip gaps of line segments
    this->connectGap();
    //cout << "line segments without gaps generation cost: " << double(clock() - start) / CLOCKS_PER_SEC << " s" << endl;
    writeImage(this->visulizeLineSeg(this->thinnedEdge), this->imageName + "_seg_lines_skipGap", false, "ed_visu");
}

EdgeSeg LineDetector::connectOneSide(bool isLeftSide, const cv::Mat &paddingThinnedEdge, const cv::Mat & paddingThinnedEdgeMask, cv::Mat &markMask, int initX, int initY, int initArgmaxO)
{
    EdgeSeg edgeSeg;
    int curX = initX;
    int curY = initY;
    int curArgmaxO = initArgmaxO;
    while(1)
    {
        vector<PosDiff> posD;
        if(isLeftSide)
            posD = this->LL.leftLinkMasks[curArgmaxO];
        else
            posD = this->LL.rightLinkMasks[curArgmaxO];
        float localV = 0;
        cv::Point neighboringP; // Adjacent continuous point
        PosDiff neighboringPointDiff; // Adjacent continuous point shift
        cv::Rect rect(curY, curX, this->LL.length * 2 + 1, this->LL.length * 2 + 1);
        cv::Mat localSearchRegion = paddingThinnedEdge(rect).clone().mul(paddingThinnedEdgeMask(rect).clone()); // local search region
        float tV; // temp value
        int mX, mY; // coordinate in markMask
        int tM; // temp mark value
        int tO; // temp orientation
        for(auto pos: posD)
        {
            tV = localSearchRegion.at<float>(pos.xDiff, pos.yDiff);
            if(tV == 0) continue; // If there is no edge, continue
            mX = curX + pos.xDiff - this->LL.length;
            mY = curY + pos.yDiff - this->LL.length;
            if(mX < 0 || mX >= markMask.rows || mY < 0 || mY >= markMask.cols) continue; // If exceed the image range, continue
            tM = markMask.at<int>(mX, mY);
            if(tM == 1) continue; // If this position has been marked, continue
            tO = this->AO.at<int>(mX, mY);
            if(computeIdxDis(tO, curArgmaxO, 12) >= 2) continue; // If the difference of two orientations are big, continue
            // If there is one edge point, and it is not been marked, and its orientationo is similar to current orientation, it is the continuous edge point. Note: the search process is interrupted
            localV = localSearchRegion.at<float>(pos.xDiff, pos.yDiff);
            neighboringPointDiff = pos;
            break;
        }
        if(localV == 0) break; // If not find the continuous edge point, break
        // Now find the continuous edge point
        neighboringP.x = curY + neighboringPointDiff.yDiff - this->LL.length;
        neighboringP.y = curX + neighboringPointDiff.xDiff - this->LL.length;
        // Apply DDA algorithm to connect the current point and the continuous point
        EdgeSeg ddaSeg = dda(curX, curY, neighboringP.y, neighboringP.x);
        // Add edge segment
        edgeSeg.extend(ddaSeg);
        // Update markMask
        for(auto p: ddaSeg.points)
            markMask.at<int>(p) = 1;
        // Update current position
        curX = neighboringP.y;
        curY = neighboringP.x;
        // 如果经历了最优朝向的跨越，更新左右侧标志
        if(curArgmaxO < this->AO.at<int>(neighboringP) && curArgmaxO <= 2 && this->AO.at<int>(neighboringP) >= 10)
            isLeftSide = !isLeftSide;
        if(this->AO.at<int>(neighboringP) < curArgmaxO && this->AO.at<int>(neighboringP) <= 2 && curArgmaxO >=10)
            isLeftSide = !isLeftSide;
        // Update current orientation
        curArgmaxO = this->AO.at<int>(neighboringP);
    }
    return edgeSeg;
}
void LineDetector::V4()
{
    int ts = this->LL.length;
    cv::Mat paddingThinnedEdge;
    cv::copyMakeBorder(this->thinnedEdge, paddingThinnedEdge, ts, ts, ts, ts, cv::BORDER_CONSTANT);
    cv::Mat paddingThinnedEdgeMask;
    cv::threshold(paddingThinnedEdge, paddingThinnedEdgeMask, 0, 1.0, cv::THRESH_BINARY);

    cv::Mat markMask = cv::Mat::zeros(this->thinnedEdge.rows, this->thinnedEdge.cols, CV_32S); // If one position is marked, its value is set to 1
    int initX, initY; // initialized position
    int initArgmaxO; // initialized orientation
    for(int i = 0; i < this->thinnedEdge.rows; i++)
    {
        for(int j = 0; j < this->thinnedEdge.cols; j++)
        {
            if(this->thinnedEdge.at<float>(i, j) == 0) continue; // If it is not the edge point, continue
            if(markMask.at<int>(i, j) == 1) continue; // If it has been marked, continue

            initX = i;
            initY = j;
            initArgmaxO = this->AO.at<int>(i, j);
            
            EdgeSeg edgeSeg;
            // Add the current position and mark it
            edgeSeg.extend(cv::Point(initY, initX));
            markMask.at<int>(i, j) = 1;
            // Connect the left side
            EdgeSeg leftPart = connectOneSide(true, paddingThinnedEdge, paddingThinnedEdgeMask, markMask, initX, initY, initArgmaxO);
            // Connect the right side
            EdgeSeg rightPart = connectOneSide(false, paddingThinnedEdge, paddingThinnedEdgeMask, markMask, initX, initY, initArgmaxO);
            // Integrate the left side and right side
            edgeSeg.extend(leftPart);
            edgeSeg.integrateRightPart(rightPart);
            // Save the continuous edge points
            if(edgeSeg.getLength() > 7)
                this->ES.push_back(edgeSeg);
        }
    }
    //cout << "ES number: " << this->ES.size() << endl;
}

void LineDetector::generateLineSeg()
{
    // For each continuous edge segment, divide it into line segments
    int edgeLength;
    int curIdx;
    for(int i = 0; i < this->ES.size(); i++)
    {
        edgeLength = ES[i].getLength();
        curIdx = 0;
        while(edgeLength - curIdx >= 2)
        {
            LineSeg ls(ES[i].points[curIdx], ES[i].points[curIdx + 1]);
            curIdx = curIdx + 2;
            // If the line segment can extend, extend it
            while(ls.canAddEndP && curIdx < edgeLength)
            {
                ls.addEndP(ES[i].points[curIdx]);
                curIdx++;
            }
            this->LS.push_back(ls);
        }
    }
}

void LineDetector::connectGap()
{
    cv::Mat logLineSegIdx = cv::Mat(this->thinnedEdge.size(), CV_32S, -1); // Log the index of line segments endpoints, if there are no endpoints, set it to -1 
    for(int i = 0; i < LS.size(); i++)
    {
        logLineSegIdx.at<int>(this->LS[i].edgeSeg.points.front()) = i;
        logLineSegIdx.at<int>(this->LS[i].edgeSeg.points.back()) = i;
    }

    int rows = this->thinnedEdge.rows;
    int cols = this->thinnedEdge.cols;
    int lineSegID;
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            lineSegID = logLineSegIdx.at<int>(i, j);
            if(lineSegID == -1) continue; // If there is no line segment, continue
            float direction; // The direction of the line segment, (0, 360]
            // If current point is the start point
            if(i == this->LS[lineSegID].edgeSeg.points[0].y && j == this->LS[lineSegID].edgeSeg.points[0].x)
                direction = computeDirection(this->LS[lineSegID].endP, this->LS[lineSegID].startP); // From the end point to the start point
            // If current point is the end point
            else
                direction = computeDirection(this->LS[lineSegID].startP, this->LS[lineSegID].endP); // From the start point to the end point
            
            //SearchRegion region(this->hparams.skipGapLength, this->hparams.skipGapWidth, direction); // search region
            int iii = int(direction) % 360;
            SearchRegion region = this->LL.directionSearchRegion[iii];
            for(auto dp: region.pDiff)
            {
                cv::Point possibleConnectedP(j + dp.yDiff, i + dp.xDiff); // The possible position to connect
                if(possibleConnectedP.y < 0 || possibleConnectedP.y >= rows || possibleConnectedP.x < 0 || possibleConnectedP.x >= cols) continue; // If exceed the image range, continue
                if(logLineSegIdx.at<int>(possibleConnectedP) == -1) continue; // If there is no other line segments, continue
                int connectedID = logLineSegIdx.at<int>(possibleConnectedP); // the possible line segment
                float orientationDiff = diffOrientation(this->LS[lineSegID].orientation, this->LS[connectedID].orientation);
                if(orientationDiff > this->hparams.diffOrientationTh) continue; // If the difference of two orientations are big, continue

                cv::Point p1 = this->LS[lineSegID].edgeSeg.points[0];
                cv::Point p2 = this->LS[lineSegID].edgeSeg.points.back();
                cv::Point p3 = this->LS[connectedID].edgeSeg.points[0];
                cv::Point p4 = this->LS[connectedID].edgeSeg.points.back();

                // Now try to connect two line segments
                LineSeg longerLineSeg = connectTwoLine(this->LS[lineSegID], this->LS[connectedID], cv::Point(j, i), possibleConnectedP);
                if(longerLineSeg.avgDis > longerLineSeg.avgDisTh) continue; // If the connected line segment has big average distance, continue
                // Replace the old line segment with the new
                this->LS[lineSegID] = longerLineSeg;
                // Update marks
                logLineSegIdx.at<int>(p1) = -1;
                logLineSegIdx.at<int>(p2) = -1;
                logLineSegIdx.at<int>(p3) = -1;
                logLineSegIdx.at<int>(p4) = -1;
                cv::Point p5 = this->LS[lineSegID].edgeSeg.points[0];
                logLineSegIdx.at<int>(p5) = lineSegID; 
                cv::Point p6 = this->LS[lineSegID].edgeSeg.points.back(); 
                logLineSegIdx.at<int>(p6) = lineSegID; 
                // Break the loop
                this->disappeardLS.push_back(this->LS[connectedID]); 
                break; 
            } 
        } 
    }

    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            lineSegID = logLineSegIdx.at<int>(i, j);
            if(lineSegID != -1)
            {
                LSWoGap.push_back(LS[lineSegID]);
            }
        }
    }
    LS.swap(LSWoGap);
    //cout << "imageName: " << this->imageName << " After skip gaps, the LineSeg size: " << LS.size() << endl;
}

float LineDetector::computeSalience(const EdgeSeg &es)
{
    float sum = 0.;
    for(auto p: es.points)
    {
        sum += this->edge.at<float>(p);
    }
    return sum / es.points.size();
}

cv::Mat LineDetector::visulizeEdgeSeg(const cv::Mat &src)
{
    vector<cv::Vec3f> colors;
    colors.push_back(cv::Vec3f(0., 0., 1.)); // 0
    colors.push_back(cv::Vec3f(0., 0.5, 1.));
    colors.push_back(cv::Vec3f(0., 1., 1.)); // 60
    colors.push_back(cv::Vec3f(0., 1., 0.5));
    colors.push_back(cv::Vec3f(0., 1., 0.)); //120
    colors.push_back(cv::Vec3f(0.5, 1., 0.));
    colors.push_back(cv::Vec3f(1., 1., 0.)); // 180
    colors.push_back(cv::Vec3f(1., 0.5, 0.));
    colors.push_back(cv::Vec3f(1., 0., 0.)); // 240
    colors.push_back(cv::Vec3f(1., 0., 0.5));
    colors.push_back(cv::Vec3f(1., 0., 1.)); // 300
    colors.push_back(cv::Vec3f(0.5, 0., 1.));

    cv::Mat result = cv::Mat::zeros(src.size(), CV_32FC3);
    int colorIdx;
    int count = 0;
    for(int i = 0; i < ES.size(); i++)
    {
        //colorIdx = rand() % colors.size(); // rand 函数在头文件 cstdlib 中
        //colorIdx = this->AO.at<int>(*(ES[i].points.begin()));
        if(i % 2 == 0)
            colorIdx = i % 12;
        else
            colorIdx = (i + 6) % 12;

        EdgeSeg tempVec = ES[i];

        if(tempVec.getLength() < 9) continue;
        else
        {
            float val = this->computeSalience(tempVec);
            for(auto p: tempVec.getPoints())
            {
                result.at<cv::Vec3f>(p) = colors[colorIdx];
            }
            count++;
        }
    }
    //cout << "satisfied edge segments' number is: " << count << endl;
    
    //cv::Vec3f gray(0.3, 0.3, 0.3);
    cv::Vec3f gray(1., 1., 1.);
    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            if(result.at<cv::Vec3f>(i, j)[0] == 0 && result.at<cv::Vec3f>(i, j)[1] == 0 && result.at<cv::Vec3f>(i, j)[2] == 0)
            {
                result.at<cv::Vec3f>(i, j) = gray;
            }
        }
    }
    return result;
}

cv::Mat LineDetector::visulizeLineSeg(const cv::Mat &src)
{
    vector<cv::Vec3f> colors;
    colors.push_back(cv::Vec3f(0., 0., 1.)); // 0
    colors.push_back(cv::Vec3f(0., 0.5, 1.));
    colors.push_back(cv::Vec3f(0., 1., 1.)); // 60
    colors.push_back(cv::Vec3f(0., 1., 0.5));
    colors.push_back(cv::Vec3f(0., 1., 0.)); //120
    colors.push_back(cv::Vec3f(0.5, 1., 0.));
    colors.push_back(cv::Vec3f(1., 1., 0.)); // 180
    colors.push_back(cv::Vec3f(1., 0.5, 0.));
    colors.push_back(cv::Vec3f(1., 0., 0.)); // 240
    colors.push_back(cv::Vec3f(1., 0., 0.5));
    colors.push_back(cv::Vec3f(1., 0., 1.)); // 300
    colors.push_back(cv::Vec3f(0.5, 0., 1.));

    cv::Mat result = cv::Mat::zeros(src.size(), CV_32FC3);
    int colorIdx;
    int count = 0;
    for(int i = 0; i < this->LS.size(); i++)
    {
        //colorIdx = rand() % colors.size(); // rand 函数在头文件 cstdlib 中
        //colorIdx = this->AO.at<int>(*(this->LS[i].edgeSeg.points.begin()));
        colorIdx = i % 12;
        EdgeSeg tempVec = LS[i].convertToEdgeSeg();
        if(LS[i].computeLength() < 15) continue;
        else
        {
            float val = this->computeSalience(tempVec);
            if(val < 0.1 && LS[i].computeLength() < 30) continue;
            for(auto p: tempVec.getPoints())
            {
            	val = 1;
                //result.at<cv::Vec3f>(p) = val * colors[colorIdx];
                result.at<cv::Vec3f>(p) = colors[colorIdx];
            }
            count++;
        }
    }
    //cout << "satisfied line segments' number is: " << count << endl;

    cv::Vec3f gray(1., 1., 1.);
    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            if(result.at<cv::Vec3f>(i, j)[0] == 0 && result.at<cv::Vec3f>(i, j)[1] == 0 && result.at<cv::Vec3f>(i, j)[2] == 0)
            {
                result.at<cv::Vec3f>(i, j) = gray;
            }
        }
    }
    return result;
}
