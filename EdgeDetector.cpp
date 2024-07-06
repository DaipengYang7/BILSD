#include "EdgeDetector.h"

cv::Mat EdgeDetector::getV1Response()
{
    return this->V1Response;
}
cv::Mat EdgeDetector::getV2ThinnedResponse()
{
    return this->V2ThinnedResponse;
}
cv::Mat EdgeDetector::getV2EstiOrient()
{
    return this->V2EstiOrient;
}
cv::Mat EdgeDetector::getColoredV2EstiOrient()
{
    return visulizeOptimalOrientation(this->V2EstiOrient, this->V2ThinnedResponse);
}
cv::Mat EdgeDetector::getColoredV2EstiOrientGrid()
{
    return visulizeOptimalOrientationGrid(this->V2EstiOrient, this->V1Response);
}

void EdgeDetector::loadImage(string fileName, const cv::Mat &src)
{
    this->imageName = fileName;
    cv::Mat tSrc;
    src.convertTo(tSrc, CV_32F, 1/255.0); // Convert to CV_32F
    // Image enhancement
    if(this->hparams.ifSqrt)
    {
	    cv::Mat tempSqrt;
	    cv::sqrt(tSrc, tempSqrt);
	    tSrc = tempSqrt;
    }
    vector<cv::Mat> bgr;
    cv::split(tSrc, bgr);
    vector<cv::Mat> parallelChannels;
    if(this->hparams.channels == 1)
    {
        parallelChannels.push_back(1 / 3. * (bgr[0] + bgr[1] + bgr[2]));
    }
    else if(this->hparams.channels == 3)
    {
        parallelChannels.push_back(1 / 3. * (bgr[0] + bgr[1] + bgr[2]));
        parallelChannels.push_back(bgr[2] - bgr[1]);
        parallelChannels.push_back(bgr[0] - 0.5 * (bgr[1] + bgr[2]));
    }

    //clock_t start = clock();

    // Edge detection
    this->V1(parallelChannels);
    // cout << "V1 cost: " << double(clock() - start) / CLOCKS_PER_SEC << " s" << endl;

    // Edge thinning and orientation estimation
    this->V2();
    //cout << "V2 cost: " << double(clock() - start) / CLOCKS_PER_SEC << " s" << endl;
}

void EdgeDetector::V1(const vector<cv::Mat> &pCh)
{

    cv::Mat gray = pCh[0].clone();
    writeImage(gray, this->imageName + "_gray", false, "ed_visu");
    vector<cv::Mat> V1OrientResponse;
    for(int i = 0; i < this->hparams.GaussianGradientKernels.size(); i++)
    {
        cv::Mat tResult;
        cv::filter2D(gray, tResult, gray.depth(), this->hparams.GaussianGradientKernels[i], cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);  // BORDER_DEFAULT = BORDER_REFLECT_101
        tResult = cv::abs(tResult);
        V1OrientResponse.push_back(tResult);
    }

    cv::Mat V1OrientColumn;
    cv::merge(V1OrientResponse, V1OrientColumn);
    cv::Mat tMaxResponse;
    tMaxResponse = findMaxOfChannels(V1OrientColumn);
    // Surround modulation
    cv::Mat surroundInhi;
    cv::filter2D(tMaxResponse, surroundInhi, tMaxResponse.depth(), this->hparams.avgInhiKernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    cv::Mat salientEdge;
    salientEdge = tMaxResponse - surroundInhi;
    cv::threshold(salientEdge, salientEdge, 0, 1.0, cv::THRESH_TOZERO); // If e < 0, then e = 0
	
    double maxVal;
    cv::minMaxLoc(salientEdge, 0, &maxVal);
    salientEdge = salientEdge.mul(float(1.0 / maxVal));
    this->V1Response = salientEdge;
    writeImage(1 - this->V1Response, this->imageName + "_V1Response", false, "ed_visu");
}

void EdgeDetector::V2()
{
    // Estimate the optimal orientation
    cv::Mat V2OrientColumn = filter2D_multiK(this->V1Response, this->hparams.orientKernels);
    this->V2Response = findMaxOfChannels(V2OrientColumn);
    this->V2EstiOrient = findArgMaxOfChannels(V2OrientColumn);
    // Prepare the values of two sides
    cv::Mat topSideEstimation, bottomSideEstimation;
    topSideEstimation = filter2D_multiK(this->V2Response, this->hparams.topSideKernels);
    bottomSideEstimation = filter2D_multiK(this->V2Response, this->hparams.bottomSideKernels);
    // Edge thinning
    this->V2ThinnedResponse = cv::Mat::zeros(this->V2Response.size(), CV_32F);
    int chns = this->hparams.numOrient_estiOri;
    int maxIdx = -1; // If there is no edge, set it to -1
    float curV = 0.0;
    float topV = 0.0;
    float bottomV = 0.0;
    for(int i = 0; i < this->V2Response.rows; i++)
    {
        for(int j = 0; j < this->V2Response.cols; j++)
        {
            maxIdx = this->V2EstiOrient.at<int>(i, j);
            if(maxIdx != -1 && this->V2Response.at<float>(i, j) > 0)
            {
                curV = this->V2Response.at<float>(i, j);
                topV = topSideEstimation.ptr<float>(i)[j * chns + maxIdx];
                bottomV = bottomSideEstimation.ptr<float>(i)[j * chns + maxIdx];
                if(curV >= topV && curV >= bottomV)
                    this->V2ThinnedResponse.at<float>(i, j) = curV;
            }
        }
    }
    
    cv::threshold(this->V2ThinnedResponse, this->V2ThinnedResponse, this->hparams.weakEdgeTh, 1.0, cv::THRESH_TOZERO);
    
    writeImage(1 - this->V2ThinnedResponse, imageName + "_V2ThinnedResponse", false, "ed_visu");
    writeImage(this->getColoredV2EstiOrient(), this->imageName + "_V2EstiOrient", false, "ed_visu");
    //writeImage(this->getColoredV2EstiOrientGrid(), this->imageName + "_edge_orientGrid", false, "ed_visu");
    // Binarize the thinned edge and visulize it
    cv::Mat t;
    cv::threshold(this->V2ThinnedResponse, t, this->hparams.weakEdgeTh, 1.0, cv::THRESH_BINARY);
    writeImage(t, this->imageName + "_V2ThinnedResponse_BINARY", false, "V2ThinnedResponseBI");
    //writeImage(visulizeOptimalOrientation(this->V2EstiOrient, t), this->imageName + "_thinnedEdge_optimalOrie", false, "ed_visu");
    //writeImage(visulizeOptimalOrientationGrid(this->V2EstiOrient, t), this->imageName + "_thinnedEdge_optimalOrie_grid", false, "ed_visu");
}
