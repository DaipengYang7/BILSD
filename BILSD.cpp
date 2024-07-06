#include <string>
#include <iostream>
#include <fstream>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "PathManager.h"
#include "EdgeDetector.h"
#include "LineDetector.h"
#include "EdgeSegment.h"

using namespace std;

int main(int argc, char **argv)
{
    //string imgsDir = "YorkUrbanDB"; // the directory of images
    //string imgsDir = "Wireframe"; // the directory of images
    string imgsDir = "testImage";
    PathManager pm(imgsDir);
    EdgeDetector ed;
    LineDetector ld;
    clock_t runningStart = clock();
    while(pm.beingCurFile)
    {
        // clock_t start = clock();
        string imgPath = pm.getCurImgPath();
        cv::Mat image = cv::imread(imgPath);
        // cout << "Now process: " << pm.getFilename() << endl;

        // Start to construct orientation columns in V1 and V2
        ed.loadImage(pm.getFilename(), image);
        // cout << "The edge detection cost: " << double(clock() - start) / CLOCKS_PER_SEC << " s" << endl;

        // Start to connect orientation-sensitive neurons and utilize LSE for line fitting
        ld.connectEdge(pm.getFilename(), ed.getV1Response(), ed.getV2ThinnedResponse(), ed.getV2EstiOrient());

        // Save the detected line segments
        string absTxtPath = "lines/" + pm.getFilename() + ".txt";
        ofstream out(absTxtPath, ios::out);
        if(out.is_open())
        {
            for(int i = 0; i < ld.LS.size(); i++)
            {
                float lineLength = ld.LS[i].computeLength();
                if(lineLength < 15) continue;
                EdgeSeg tempVec = ld.LS[i].convertToEdgeSeg();
                float val = ld.computeSalience(tempVec);
                if(val < 0.1 && lineLength < 30) continue; // std: 0.1 30, for OriginalYorkUrban: 0.15 50
                out << float(ld.LS[i].startP.x) << " " << float(ld.LS[i].startP.y) << " " << float(ld.LS[i].endP.x) << " " << float(ld.LS[i].endP.y) << endl; // cv::Point: x -> cols, y -> rows
            }
            out.close();
        }

        // cout << "The line segment detection cost: " << double(clock() - start) / CLOCKS_PER_SEC << " s" << endl;
        // Get the information of the next image
        pm.nextFile();
    }
    cout << "The average running time is " << double(clock() - runningStart) / CLOCKS_PER_SEC / pm.getImagesNumber()  << " s" << endl;
}
