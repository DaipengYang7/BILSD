#include "utils.h"

void rmAndMkdir(fs::path fileDir, bool isRemove)
{
    // If the fileDir is BEING and it is need to remove, remove the old and create the new
    if(fs::exists(fileDir) && isRemove)
    {
        fs::remove_all(fileDir);
        fs::create_directory(fileDir);
    }
    // If the fileDir is BEING and it is not need to remove, do nothing
    else if(fs::exists(fileDir) && !isRemove) {}
    // If the fileDir is not BEING, create it
    else
        fs::create_directory(fileDir);
}
float findMaxValOfMat(const cv::Mat &src)
{
    // Find the maximum value of the mat with single-channel or 3-channels
    double maxVal;
    if(src.depth() == CV_32F && src.channels() == 1)
    {
        cv::minMaxLoc(src, 0, &maxVal);
        return float(maxVal);
    }
    else if(src.depth() == CV_32F && src.channels() == 3)
    {
        vector<cv::Mat> matVec;
        cv::split(src, matVec);
        double tMaxVal;
        for(int i = 0; i < src.channels(); i++)
        {
            cv::minMaxLoc(matVec[i], 0, &tMaxVal);
            if(maxVal < tMaxVal)
                maxVal = tMaxVal;
        }
        return float(maxVal);
    }
    else
    {
        return 0.0;
    }
}
void writeImage(const cv::Mat &image, const string &name, bool maxValTo1, string dirname)
{
    fs::path curFileDir = fs::current_path(); // Get the directory of current file
    fs::path savedDir = curFileDir / dirname; // The directory for saved file
    rmAndMkdir(savedDir, false);
    fs::path savedPath = savedDir / (name + ".png"); // The saved path for image
#if TEST_UTILS
    cout << "Current file directory: " << curFileDir.string() << endl;
    cout << "Created directory: " << savedDir << endl;
    cout << "Saved image path: " << savedPath.string() << endl;
#endif

    cv::Mat savedImage;
    // 1. If image.depth() == CV_8U, no format coversion
    if(image.depth() == CV_8U && (image.channels() == 1 || image.channels() == 3))
    {
        savedImage = image;
        cv::imwrite(savedPath, savedImage);
    }
    // 2. If image.depth() == CV_32F, format conversion
    else if(image.depth() == CV_32F && (image.channels() == 1 || image.channels() == 3))
    {
        if(maxValTo1)
        {
            float maxVal = findMaxValOfMat(image);
            image.convertTo(savedImage, CV_8U, 255 * 1 / (maxVal + 1e-4));
        }
        else
            image.convertTo(savedImage, CV_8U, 255);
        cv::imwrite(savedPath, savedImage);
    }
    // 3. Special case: visulize the image mask
    else if(image.depth() == CV_32S && image.channels() == 1)
    {
        image.convertTo(savedImage, CV_8U, 255);
        cv::imwrite(savedPath, savedImage);
    }
    // 4. Special case: visulize the multi-channel mat
    else if(image.depth() == CV_32F && image.channels() > 3)
    {
        cv::Mat savedImage;
        vector<cv::Mat> multiChannels;
        cv::split(image, multiChannels);
        for(int i = 0; i < multiChannels.size(); i++)
        {
            savedPath = savedDir / (name + "_" + to_string(i) + ".png");
            multiChannels[i].convertTo(savedImage, CV_8U, 255);
            cv::imwrite(savedPath, savedImage);
        }
    }
    else
        cout << "The format of image is not right, please check it." << endl;
}

int computeIdxDis(int i1, int i2, int mod)
{
    if(i1 < i2)
        return min(i2 - i1, (i1 + mod - i2) % mod);
    else
        return min(i1 - i2, (i2 + mod - i1) % mod);
}

float computeDirection(double dx, double dy)
{
    // Compute the angle between the line direction and X-axis, A \in (0, 360], counterclockwise rotation
    float direction = std::atan2(dy, dx) / CV_PI * 180; // atan2(double y, double x) y -> Y-axis, x -> X-axis, A \in (-pi, pi], be symmetric with respect to X-axis
    if(direction < 0)
        direction += 360;
    return direction;
}
float computeDirection(const cv::Point2d &p1, const cv::Point2d &p2)
{
    double dx = p2.y - p1.y;
    double dy = p2.x - p1.x;
    return computeDirection(dx, dy);
}
float diffDirection(float d1, float d2)
{
    if(d1 < d2)
        return min(d2 - d1, d1 + 360 -d2);
    else
        return min(d1 - d2, d2 + 360 - d1);
}
float computeOrientation(double x)
{   
    // Compute the angle between the line orientation and X-axis, A \in (0, 180], counterclockwise rotation
    float orientation = std::atan(x) / CV_PI * 180; // atan(double x) \in (-pi/2, pi/2], be symmetric with respect to X-axis
    if(orientation < 0)
        orientation += 180;
    return orientation;
}
float diffOrientation(float o1, float o2)
{
    if(o1 < o2)
        return min(o2 - o1, o1 + 180 -o2);
    else
        return min(o1 - o2, o2 + 180 - o1);
}

cv::Mat filter2D_multiK(const cv::Mat &src, const vector<cv::Mat> &kernels)
{
    vector<cv::Mat> resultVec;
    for(int i = 0; i < kernels.size(); i++)
    {
        cv::Mat temp;
        cv::filter2D(src, temp, src.depth(), kernels[i], cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        resultVec.push_back(temp);
    }
    cv::Mat result;
    cv::merge(resultVec, result);
#if TEST_UTILS
    cout << "TEST_UTILS The merged result own " << result.channels() << " channels" << endl;
#endif
    return result;
}
cv::Mat findArgMaxOfChannels(const cv::Mat &src)
{
    int chns = src.channels();
    // Only consider the mat with multiple channels
    if(chns <= 1)
    {
        cout << "Please check your input, whose channels are not expected." << endl;
        return cv::Mat::zeros(src.size(), CV_8U);
    }
    // Only consider the FLOAT type data
    if(src.depth() != CV_32F)
    {
        cout << "Please check your input, whose depth is not expected." << endl;
        return cv::Mat::zeros(src.size(), CV_8U);
    }

    cv::Mat argMaxIdx = cv::Mat::zeros(src.size(), CV_32S);
    float curVal = 0.;
    float maxVal = 0.;
    int maxIdx = -1; // If there is no edge, assign it with -1
    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            curVal = 0.;
            maxVal = 0.;
            maxIdx = -1;
            for(int k = 0; k < chns; k++)
            {
                curVal = src.ptr<float>(i)[j * chns + k];
                if(curVal != 0 && curVal >= maxVal)
                {
                    maxVal = curVal;
                    maxIdx = k;
                }
            }
            argMaxIdx.at<int>(i, j) = maxIdx;
        }
    }
    return argMaxIdx;
}
cv::Mat findMaxOfChannels(const cv::Mat &src)
{
    int chns = src.channels();
    // Only consider the mat with multiple channels
    if(chns <= 1)
    {
        cout << "Please check your input, whose channels are not expected." << endl;
        return cv::Mat::zeros(src.size(), CV_8U);
    }
    // Only consider the FLOAT type data
    if(src.depth() != CV_32F)
    {
        cout << "Please check your input, whose depth is not expected." << endl;
        return cv::Mat::zeros(src.size(), CV_8U);
    }

    cv::Mat result = cv::Mat::zeros(src.size(), CV_32F);
    float curVal = 0.;
    float maxVal = 0.;
    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            curVal = 0.;
            maxVal = 0.;
            for(int k = 0; k < chns; k++)
            {
                curVal = src.ptr<float>(i)[j * chns + k];
                if(curVal != 0 && curVal >= maxVal)
                    maxVal = curVal;
            }
            result.at<float>(i, j) = maxVal;
        }
    }
    return result;
}



cv::Mat visulizeOptimalOrientation(const cv::Mat &estiOptimalOrient, const cv::Mat &edgeVal)
{
    cv::Vec3f black(0., 0., 0.); // If there is no edge, black
    cv::Vec3f white(1., 1., 1.); // If there is no edge, white
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

    cv::Mat result(estiOptimalOrient.size(), CV_32FC3);
    for(int i = 0; i < estiOptimalOrient.rows; i++)
    {
        for(int j = 0; j < estiOptimalOrient.cols; j++)
        {
            int idx = estiOptimalOrient.at<int>(i, j);
            if(idx == -1 || edgeVal.at<float>(i, j) == 0)
            {
                //result.at<cv::Vec3f>(i, j) = black;
                result.at<cv::Vec3f>(i, j) = white;
            }
            else
            {
                //result.at<cv::Vec3f>(i, j) = colors[idx] * edgeVal.at<float>(i, j);
                //result.at<cv::Vec3f>(i, j) = colors[idx];
                result.at<cv::Vec3f>(i, j) = colors[idx] * (edgeVal.at<float>(i, j));
            }
        }
    }
    return result;
}
cv::Mat visulizeOptimalOrientationGrid(const cv::Mat &estiOptimalOrient, const cv::Mat &edgeVal)
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
    vector<cv::Mat> lineTemplates;
    for(int i = 0; i < 12; i++)
    {
        cv::Mat temp = cv::Mat::zeros(5, 5, CV_8U);
        temp.at<uchar>(2, 2) = 1;
        switch(i)
        {
            case 0:
                temp.at<uchar>(0, 2) = 1;
                temp.at<uchar>(1, 2) = 1;
                temp.at<uchar>(3, 2) = 1;
                temp.at<uchar>(4, 2) = 1;
                break;
            case 1:
                temp.at<uchar>(0, 3) = 1;
                temp.at<uchar>(1, 2) = 1;
                temp.at<uchar>(3, 2) = 1;
                temp.at<uchar>(4, 1) = 1;
                break;
            case 2:
                temp.at<uchar>(0, 3) = 1;
                temp.at<uchar>(1, 3) = 1;
                temp.at<uchar>(3, 1) = 1;
                temp.at<uchar>(4, 1) = 1;
                break;
            case 3:
                temp.at<uchar>(0, 4) = 1;
                temp.at<uchar>(1, 3) = 1;
                temp.at<uchar>(3, 1) = 1;
                temp.at<uchar>(4, 0) = 1;
                break;
            case 4:
                temp.at<uchar>(1, 4) = 1;
                temp.at<uchar>(1, 3) = 1;
                temp.at<uchar>(3, 1) = 1;
                temp.at<uchar>(3, 0) = 1;
                break;
            case 5:
                temp.at<uchar>(1, 4) = 1;
                temp.at<uchar>(2, 3) = 1;
                temp.at<uchar>(2, 1) = 1;
                temp.at<uchar>(3, 0) = 1;
                break;
            case 6:
                temp.at<uchar>(2, 0) = 1;
                temp.at<uchar>(2, 1) = 1;
                temp.at<uchar>(2, 3) = 1;
                temp.at<uchar>(2, 4) = 1;
                break;
            case 7:
                temp.at<uchar>(1, 0) = 1;
                temp.at<uchar>(2, 1) = 1;
                temp.at<uchar>(2, 3) = 1;
                temp.at<uchar>(3, 4) = 1;
                break;
            case 8:
                temp.at<uchar>(1, 0) = 1;
                temp.at<uchar>(1, 1) = 1;
                temp.at<uchar>(3, 3) = 1;
                temp.at<uchar>(3, 4) = 1;
                break;
            case 9:
                temp.at<uchar>(0, 0) = 1;
                temp.at<uchar>(1, 1) = 1;
                temp.at<uchar>(3, 3) = 1;
                temp.at<uchar>(4, 4) = 1;
                break;
            case 10:
                temp.at<uchar>(0, 1) = 1;
                temp.at<uchar>(1, 1) = 1;
                temp.at<uchar>(3, 3) = 1;
                temp.at<uchar>(4, 3) = 1;
                break;
            case 11:
                temp.at<uchar>(0, 1) = 1;
                temp.at<uchar>(1, 2) = 1;
                temp.at<uchar>(3, 2) = 1;
                temp.at<uchar>(4, 3) = 1;
                break;
        }
        lineTemplates.push_back(temp);
    }
    // Combine color with local line templates
    vector<cv::Mat> coloredLineTemplates;
    for(int i = 0; i < 12; i++)
    {
        cv::Mat temp = cv::Mat::zeros(5, 5, CV_32FC3);
        for(int ki = 0; ki < 5; ki++)
        {
            for(int kj = 0; kj < 5; kj++)
            {
                temp.at<cv::Vec3f>(ki, kj) = 
            lineTemplates[i].at<uchar>(ki, kj) * colors[i];
            }
        }
        coloredLineTemplates.push_back(temp);
    }

    // Test the colored local line template
    // for(int i = 0; i < 12; i++)
    //     writeImage(coloredLineTemplates[i], "coloredLineTemplate_" + to_string(i) + ".png", false);

    cv::Mat result = cv::Mat::zeros(estiOptimalOrient.rows * 5, estiOptimalOrient.cols * 5, CV_32FC3);
    for(int i = 0; i < estiOptimalOrient.rows; i++)
    {
        for(int j = 0; j < estiOptimalOrient.cols; j++)
        {
            int idx = estiOptimalOrient.at<int>(i, j);
            if(idx == -1) continue; // If no edge, do nothing
            else // If there is BEING edge, replace the black block with the colored local line template
            {
                for(int ti = i * 5, ki = 0; ti < (i + 1) * 5; ti++, ki++)
                {
                    for(int tj = j * 5, kj = 0; tj < (j + 1) * 5; tj++, kj++)
                    {
                        result.at<cv::Vec3f>(ti, tj) = coloredLineTemplates[idx].at<cv::Vec3f>(ki, kj) * edgeVal.at<float>(i, j);
                    }
                }
            }
        }
    }
    return result;
}
