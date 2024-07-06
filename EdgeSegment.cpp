#include "EdgeSegment.h"

EdgeSeg dda(int h1, int w1, int h2, int w2)
{
    // A naive way of drawing line by utilizing DDA algorithm
    vector<cv::Point> edgePoints;
    // If these tow points are same, add it and return
    if(w1 == w2 && h1 == h2)
    {
        edgePoints.push_back(cv::Point(w1, h1));
        return edgePoints;
    }
    // If these two points are neighbors in 8-neighbors
    if(abs(h2 - h1) <= 1 && abs(w2 - w1) <= 1)
    {
        edgePoints.push_back(cv::Point(w1, h1));
        edgePoints.push_back(cv::Point(w2, h2));
        return edgePoints;
    }
    // Apply the DDA algorithm
    float dw = w2 - w1;
    float dh = h2 - h1;
    // Slope
    float steps = max(abs(dh), abs(dw));
    // One is equal to 1, and another less than 1
    float delta_w = dw / steps;
    float delta_h = dh / steps;
    // Round to the nearest, ensuring that the increments of 'w' and 'h' are less than or equal to 1, to make the generated lines as evenly distributed as possible
    float w = w1 + 0.5;
    float h = h1 + 0.5;
    for(int i = 0; i < int(steps + 1); i++)
    {
        // Add points
        edgePoints.push_back(cv::Point(int(w), int(h)));
        w += delta_w;
        h += delta_h;
    }
    // Note that the DDA algorithm might reorder the points on the edge and requires rearrangement.
    if(edgePoints[0].x != w1 || edgePoints[0].y != h1)
        reverse(edgePoints.begin(), edgePoints.end());
    return EdgeSeg(edgePoints);
}
EdgeSeg dda(double h1, double w1, double h2, double w2)
{
    // A naive way of drawing line by utilizing DDA algorithm
    vector<cv::Point> edgePoints;
    // If these tow points are same, add it and return
    if(w1 == w2 && h1 == h2)
    {
        edgePoints.push_back(cv::Point(int(w1 + 0.5), int(h1 + 0.5)));
        return edgePoints;
    }
    // If these two points are neighbors in 8-neighbors
    if(abs(h2 - h1) <= 1 && abs(w2 - w1) <= 1)
    {
        edgePoints.push_back(cv::Point(int(w1 + 0.5), int(h1 + 0.5)));
        edgePoints.push_back(cv::Point(int(w2 + 0.5), int(h2 + 0.5)));
        return edgePoints;
    }
    // Apply the DDA algorithm
    float dw = w2 - w1;
    float dh = h2 - h1;
    // Slope
    float steps = max(abs(dh), abs(dw));
    // One is equal to 1, and another less than 1
    float delta_w = dw / steps;
    float delta_h = dh / steps;
    // Round to the nearest, ensuring that the increments of 'w' and 'h' are less than or equal to 1, to make the generated lines as evenly distributed as possible
    float w = w1 + 0.5;
    float h = h1 + 0.5;
    for(int i = 0; i < int(steps + 1); i++)
    {
        // Add points
        edgePoints.push_back(cv::Point(int(w), int(h)));
        w += delta_w;
        h += delta_h;
    }
    // Note that the DDA algorithm might reorder the points on the edge and requires rearrangement.
    if(edgePoints[0].x != w1 || edgePoints[0].y != h1)
        reverse(edgePoints.begin(), edgePoints.end());
    return EdgeSeg(edgePoints);
}
EdgeSeg dda(cv::Point p1, cv::Point p2)
{
    return dda(p1.y, p1.x, p2.y, p2.x);
}
EdgeSeg dda(cv::Point2d p1, cv::Point2d p2)
{
    return dda(p1.y, p1.x, p2.y, p2.x);
}
