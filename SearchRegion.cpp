#include "SearchRegion.h"

bool cmp_PosDiff(PosDiff pd1, PosDiff pd2)
{
   return pd1.xDiff * pd1.xDiff + pd1.yDiff * pd1.yDiff < pd2.xDiff * pd2.xDiff + pd2.yDiff * pd2.yDiff;
}

SearchRegion::SearchRegion(int length, int width, float direction)
{
    this->length = length;
    this->width = width;
    this->direction = direction;
    this->constructRegion();
}

void SearchRegion::constructRegion()
{
    int searchHW = this->length * 2 + 1;
    int centre = this->length;
    float halfWidth = float(this->width) / 2.;

    // line: y = tan(direction) * x  --> sin(d)x - cos(d)y = 0
    // line equation: A*x + B*y + C = 0
    float theta = direction / 180 * CV_PI;
    float A, B, C;
    A = sin(theta);
    B = - cos(theta);
    C = - (A * centre + B * centre);

    float shiftDis = 10.0; // The shift for determining the left or right sides
    float shiftX = centre + cos(theta + CV_PI / 2) * shiftDis;
    float shiftY = centre + sin(theta + CV_PI / 2) * shiftDis;

    float cpX, cpY;
    float disToLine, disToLineCentre;
    for(int i = 0; i < searchHW; i++)
    {
        for(int j = 0; j < searchHW; j++)
        {
            // Compute the distance to the line
            // d = (A * x0 + B * y0 + C) / sqrt(A**2 + B**2)
            disToLine = abs(A * i + B * j + C);
            // Get the prejection on the line
            // x = (B**2 * x0 - A * B * y0 - A * C) / (A**2 + B**2)
            // y = (- A * B * x0 + A**2 * y0 - B * C) / (A**2 + B**2)
            cpX = (B * B * i - A * B * j - A * C);
            cpY = (- A * B * i + A * A * j - B * C);
            // Compute the distance between "the prejection point" and "the middle point of the line"
            disToLineCentre = sqrt(pow(cpX - centre, 2) + pow(cpY - centre, 2));

            if(disToLineCentre <= this->length && disToLine <= halfWidth)
            {
                if(this->_isRight(float(centre), float(centre), float(shiftX), float(shiftY), float(i), float(j)))
                    pDiff.push_back(PosDiff(i - this->length, j - this->length));
            }
        }
    }
    sort(this->pDiff.begin(), this->pDiff.end(), cmp_PosDiff); 
}

bool SearchRegion::_isLeft(float Ax, float Ay, float Bx, float By, float Px, float Py)
{
    return (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax) > 1e-4;
}
bool SearchRegion::_isRight(float Ax, float Ay, float Bx, float By, float Px, float Py)
{
    return (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax) < -1e-4;
}
