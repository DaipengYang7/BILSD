#include "LineSegment.h"

LineSeg::LineSeg(const cv::Point &p1, const cv::Point &p2)
{
    this->A = this->B = this->C = this->D = this->E = 0;
    this->lineCo[0] = this->lineCo[1] = this->lineCo[2] = 0.;
    this->addFirstTwoPoint(p1, p2);
    this->update();
}

void LineSeg::addFirstTwoPoint(const cv::Point &p1, const cv::Point &p2)
{
    this->edgeSeg.extend(p1);
    this->edgeSeg.extend(p2);
    this->A += p1.y * p1.y + p2.y * p2.y;
    this->B += p1.y + p2.y;
    this->C += p1.y * p1.x + p2.y * p2.x;
    this->D += p1.x + p2.x;
    this->E += p1.x * p1.x + p2.x * p2.x;
}

void LineSeg::addEndP(cv::Point &p)
{
    if(this->canAddEndP) 
    {
        this->edgeSeg.extend(p);
        this->A += p.y * p.y; // Note that cv::Point x->cols y->rows
        this->B += p.y;
        this->C += p.y * p.x;
        this->D += p.x;
        this->E += p.x * p.x;
        this->update();
    }
}

void LineSeg::removeEndP()
{
    if(this->edgeSeg.getLength() >= 3)
    {
        cv::Point p;
        p = this->edgeSeg.remove();
        this->A -= p.y * p.y; // Note that cv::Point x->cols y->rows
        this->B -= p.y;
        this->C -= p.y * p.x;
        this->D -= p.x;
        this->E -= p.x * p.x;
        this->update();
    }
}

void LineSeg::LSE()
{
    // Ordinary least squares estimation method for line fit
    int n = this->edgeSeg.getLength();
    if(!isInverseXY)
    {
        // y = ax + b --> rx + sy + t = 0
        // rx + sy + t = 0 --> r = (nC - BD) | s = (B^2 - nA) | t = (DA - BC)
        lineCo[0] = (n * C - B * D);
        lineCo[1] = (B * B - n * A);
        lineCo[2] = (D * A - B * C);
    }
    else
    {
        // y = ax + b --> rx + sy + t = 0 (coordinate system conversion)
        // A->E | B->D | C->C | D->B | E->A
        lineCo[0] = (n * C - D * B);
        lineCo[1] = (D * D - n * E);
        lineCo[2] = (B * E - D * C);
    }
    float normalizedFactor = sqrt(lineCo[0] * lineCo[0] + lineCo[1] * lineCo[1]);
    lineCo[0] /= normalizedFactor;
    lineCo[1] /= normalizedFactor;
    lineCo[2] /= normalizedFactor;
}

void LineSeg::getEndpoints()
{
    cv::Point p1, p2;
    p1 = this->edgeSeg.getFront();
    p2 = this->edgeSeg.getBack();
    float x1, y1, x2, y2;
    if(!isInverseXY)
    {
        // 已知直线方程 c_0x + c_1y + c_2 = 0 (c_0^2 + c_1^2 = 1)，点(x0, y0)
        // 则投影为
        // x = c_1^2x_0 - c_0c_1y_0 - c_0c_2
        // y = c_0^2y_0 - c_0c_1x_0 - c_1c_2
        // 注意：1. x1, y1, x2, y2 是正常的图像行标，列标
        //       2. Point 行标列标顺序要调换一下
        x1 = this->lineCo[1] * this->lineCo[1] * p1.y - this->lineCo[0] * this->lineCo[1] * p1.x - this->lineCo[0] * this->lineCo[2];
        y1 = this->lineCo[0] * this->lineCo[0] * p1.x - this->lineCo[0] * this->lineCo[1] * p1.y - this->lineCo[1] * this->lineCo[2];
        x2 = this->lineCo[1] * this->lineCo[1] * p2.y - this->lineCo[0] * this->lineCo[1] * p2.x - this->lineCo[0] * this->lineCo[2];
        y2 = this->lineCo[0] * this->lineCo[0] * p2.x - this->lineCo[0] * this->lineCo[1] * p2.y - this->lineCo[1] * this->lineCo[2];
        this->startP.x = y1;
        this->startP.y = x1;
        this->endP.x = y2;
        this->endP.y = x2;
    }
    else
    {
        // 已知直线方程 c_0x + c_1y + c_2 = 0 (c_0^2 + c_1^2 = 1)，点(x0, y0)
        // 则投影为
        // x = c_1^2x_0 - c_0c_1y_0 - c_0c_2
        // y = c_0^2y_0 - c_0c_1x_0 - c_1c_2
        // 注意：1. x1, y1, x2, y2 是在 (XOY XY 调换后) 的正常的图像行标，列标，即对应图像：输入的 x 为列标，y 为行标，然后输出的 x1 x2 为 列标，y1 y2 为行标
        //       2. Point 的 x 为列标，y 为行标
        x1 = this->lineCo[1] * this->lineCo[1] * p1.x - this->lineCo[0] * this->lineCo[1] * p1.y - this->lineCo[0] * this->lineCo[2];
        y1 = this->lineCo[0] * this->lineCo[0] * p1.y - this->lineCo[0] * this->lineCo[1] * p1.x - this->lineCo[1] * this->lineCo[2];
        x2 = this->lineCo[1] * this->lineCo[1] * p2.x - this->lineCo[0] * this->lineCo[1] * p2.y - this->lineCo[0] * this->lineCo[2];
        y2 = this->lineCo[0] * this->lineCo[0] * p2.y - this->lineCo[0] * this->lineCo[1] * p2.x - this->lineCo[1] * this->lineCo[2];
        this->startP.x = x1;
        this->startP.y = y1;
        this->endP.x = x2;
        this->endP.y = y2;
    }
}

float LineSeg::computeAvgDis()
{
    this->totalDis = 0.;
    double curDis = 0.;
    for(auto p: this->edgeSeg.points)
    {
        if(!this->isInverseXY)
            curDis = abs((lineCo[0] * p.y + lineCo[1] * p.x + lineCo[2]));
        else
            curDis = abs((lineCo[0] * p.x + lineCo[1] * p.y + lineCo[2]));
        this->totalDis += curDis;
    }
    this->avgDis = this->totalDis / this->edgeSeg.getLength();
    return curDis;
}

void LineSeg::update()
{
    float dw = this->edgeSeg.getBack().x - this->edgeSeg.getFront().x;
    float dh = this->edgeSeg.getBack().y - this->edgeSeg.getFront().y;
    this->orientation = computeOrientation(dh / dw);

    if(abs(dw) > abs(dh))
        isInverseXY = true;
    else
        isInverseXY = false;

    this->LSE();
    this->getEndpoints();
    float curDis = this->computeAvgDis();
    if(this->avgDis >= this->avgDisTh || curDis > this->curDisTh)
    {
        this->canAddEndP = false;
        this->removeEndP();
    }

    //cout << "*****************" << endl;
    //cout << "edgeSeg:" << endl;
    //cout << "p.x: " << edgeSeg.getFront().x << " | p.y: " << edgeSeg.getFront().y << endl;
    //cout << "p.x: " << edgeSeg.getBack().x << " | p.y: " << edgeSeg.getBack().y << endl;
    //cout << "startP: " << startP.x << " , " << startP.y << endl;
    //cout << "endP: " << endP.x << " , " << endP.y << endl;
    //cout << "n points: " << edgeSeg.getLength() << endl;
    //cout << "totalDis: " << totalDis << " | avgDis: " << avgDis << endl;
    //cout << "*****************" << endl;
}

LineSeg connectTwoLine(LineSeg ls1, LineSeg ls2, cv::Point p1, cv::Point p2)
{
    // Adjust ls1 and ls2: the end point of ls1 and the start point of ls2 are connected
    // If the start point of ls2 is the connection point, reverse
    if(ls1.edgeSeg.points[0].x == p1.x && ls1.edgeSeg.points[0].y == p1.y)
    {
        // Reverse edge points
        reverse(ls1.edgeSeg.points.begin(), ls1.edgeSeg.points.end()); // algorithm
        // Swap start point and end point 
        cv::Point2d tP; 
        tP = ls1.endP;
        ls1.endP = ls1.startP;
        ls1.startP = tP;
    }
    // If the start point of ls2 is not the connection point, reverse
    if(ls2.edgeSeg.points[0].x != p2.x || ls2.edgeSeg.points[0].y != p2.y)
    {
        // Reverse edge points
        reverse(ls2.edgeSeg.points.begin(), ls2.edgeSeg.points.end()); // algorithm
        // Swap start point and end point 
        cv::Point2d tP; 
        tP = ls2.endP;
        ls2.endP = ls2.startP;
        ls2.startP = tP;
    }

    // Connect two line segments
    LineSeg curLS = ls1;
    curLS.edgeSeg.extend(ls2.edgeSeg);
    curLS.A += ls2.A;
    curLS.B += ls2.B;
    curLS.C += ls2.C;
    curLS.D += ls2.D;
    curLS.E += ls2.E;

    float dw = curLS.edgeSeg.getBack().x - curLS.edgeSeg.getFront().x;
    float dh = curLS.edgeSeg.getBack().y - curLS.edgeSeg.getFront().y;
    curLS.orientation = computeOrientation(dh / dw);
    if(abs(dw) > abs(dh))
        curLS.isInverseXY = true;
    else
        curLS.isInverseXY = false;
    curLS.LSE();
    curLS.getEndpoints();
    curLS.computeAvgDis();
    return curLS;
}
