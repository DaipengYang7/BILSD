#include "EdgeDetectorHyperparams.h"

cv::Mat sumOfKernelTo1(const cv::Mat kernel)
{
    return kernel / cv::sum(kernel)[0];
}
cv::Mat equalKernelWeigth(const cv::Mat &kernel)
{
    // Set the positive weights and the absolute values of negative weights to equal to 1
    float sumP = 0.;
    float sumN = 0.;
    cv::Mat temp = kernel.clone();
    for(int i = 0; i < temp.rows; i++)
    {
        for(int j = 0; j < temp.cols; j++)
        {
            if(temp.at<float>(i, j) > 0)
                sumP += temp.at<float>(i, j);
            if(temp.at<float>(i, j) < 0)
                sumN += temp.at<float>(i, j);
        }
    }
    for(int i = 0; i < temp.rows; i++)
    {
        for(int j = 0; j < temp.cols; j++)
        {
            if(temp.at<float>(i, j) > 0)
                temp.at<float>(i, j) /= sumP;
            if(temp.at<float>(i, j) < 0)
                temp.at<float>(i, j) /= -sumN;
        }
    }
    return temp;
}
cv::Mat getAvgKernel(int size)
{
    cv::Mat kernel = cv::Mat::ones(size, size, CV_32F);
    float centre = float(size / 2);
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
            float dis = sqrt((i - centre) * (i - centre) + (j - centre) * (j - centre));
            if(dis > centre)
                kernel.at<float>(i, j) = 0;
        }
    }
    return kernel / cv::sum(kernel)[0];
}
cv::Mat zeroKernelCentre(cv::Mat kernel, float centreRadius)
{
    // Only consider kernel.depth() == CV_32F
    int centre = kernel.rows / 2;
    cv::Mat result = cv::Mat::zeros(kernel.size(), CV_32F);
    for(int i = 0; i < kernel.rows; i++)
    {
        for(int j = 0; j < kernel.cols; j++)
        {
            if((i - centre) * (i - centre) + (j - centre) * (j - centre) <= centreRadius * centreRadius)
            {
                result.at<float>(i, j) = 0;
            }
            else
                result.at<float>(i, j) = kernel.at<float>(i, j);
        }
    }
    return result;
}
cv::Mat getKernelWithRadiusConstraint(cv::Mat kernel, int radius, bool isSumTo1)
{
    cv::Mat result;
    int halfKernelSize= kernel.rows / 2;
    if(halfKernelSize <= radius)
    {
        result = kernel;
    }
    else
    {
        for(int i = 0; i < kernel.rows; i++)
        {
            for(int j = 0; j < kernel.cols; j++)
            {
                int dx = i - halfKernelSize;
                int dy = j - halfKernelSize;
                if(dx * dx + dy * dy > radius * radius)
                    kernel.at<float>(i, j) = 0;
            }
        }
        cv::Rect roiRect(halfKernelSize - radius, halfKernelSize - radius, 2 * radius + 1, 2 * radius + 1);
        result = kernel(roiRect).clone();
    }
    if(isSumTo1) // Set the sum of kernel to 1 according to the need
        result = sumOfKernelTo1(result);
    return result;
}
cv::Mat discardTinyWeight(cv::Mat kernel, float ratio)
{
    // Discard the tiny values of kernel
    double maxV;
    cv::minMaxLoc(kernel, 0, &maxV);
    maxV += 1e-4;
    for(int i = 0; i < kernel.rows; i++)
    {
        for(int j = 0; j < kernel.cols; j++)
        {
            if(abs(kernel.at<float>(i, j) / maxV) < ratio)
                kernel.at<float>(i, j) = 0;
        }
    }
    return kernel;
}
int computeGaussianSize(double sigma)
{
    double th = 1e-2;
    double k = 2 * sigma * sigma;
    int radius = int(sqrt(-k * log(th * sqrt(k * CV_PI))));
    return 2 * radius + 1;
}
cv::Mat getGaussianGradientKernel(float sigma, float seta, float theta)
{
    int ks = computeGaussianSize(sigma);
    cv::Mat kernel(ks, ks, CV_32F);

    int center = ks / 2;
    float disX, disY, x, y;
    for(int i = 0; i < ks; i++)
    {
        for(int j = 0; j < ks; j++)
        {
            disX = j - center;
            disY = i - center;
            x = disX * cos(theta) + disY * sin(theta);
            y = - disX * sin(theta) + disY * cos(theta);
            kernel.at<float>(i, j) = - x * exp(- (x * x + seta * seta * y * y) / (2 * sigma * sigma)) / (CV_PI * sigma * sigma);
        }
    }
    return kernel;
}
cv::Mat getEllipseGaussianKernel(float sigma_x, float sigma_y, float theta)
{
    int ks = computeGaussianSize(max(sigma_x, sigma_y));
    cv::Mat kernel(ks, ks, CV_32F);

    float centre_x = ks / 2;
    float centre_y = ks / 2;
    float a = pow(cos(theta), 2) / (2 * sigma_x * sigma_x) + pow(sin(theta), 2) / (2 * sigma_y * sigma_y);
    float b = -sin(2 * theta) / (4 * sigma_x * sigma_x) + sin(2 * theta) / (4 * sigma_y * sigma_y);
    float c = pow(sin(theta), 2) / (2 * sigma_x * sigma_x) + pow(cos(theta), 2) / (2 * sigma_y * sigma_y);

    for(int i = 0; i < ks; i ++)
    {
        for(int j = 0; j < ks; j++)
        {
            float x = i - centre_x;
            float y = j - centre_y;
            float t = exp(- (a * pow(x, 2) + 2 * b * x * y + c * pow(y, 2)));
            kernel.at<float>(i, j) = t;
        }
    }
    kernel = kernel / cv::sum(kernel)[0];
    return kernel;
}
cv::Mat getGaborKernel(int ksize, float sigma, float theta, float lambda, float gamma, float psi)
{
    cv::Mat kernel(ksize, ksize, CV_32F);
    float sigmaX = sigma;
    float sigmaY = sigma / gamma;
    float c = cos(theta);
    float s = sin(theta);
    // 确定卷积核函数的变化范围
    int xmax = ksize / 2;
    int ymax = ksize / 2;
    int xmin = -xmax;
    int ymin = -ymax;

    float ex = -0.5 / (sigmaX * sigmaX);
    float ey = -0.5 / (sigmaY * sigmaY);
    float cscale =  CV_PI * 2 / lambda;

    float _x = 0;
    float _y = 0;
    for(int y = ymin; y <= ymax; y++)
    {
        for(int x = xmin; x <= xmax; x++)
        {
            _x = x * c + y * s;
            _y = -x * s + y * c;
            float val = exp(ex * _x * _x + ey * _y * _y) * cos(cscale * _x + psi);
            kernel.at<float>(ymax + y, xmax + x) = val;
        }
    }
    return kernel;
}

void getTBsideKernels(int i, cv::Mat &topK, cv::Mat &bottomK)
{
    topK = cv::Mat::zeros(3, 3, CV_32F);
    bottomK = cv::Mat::zeros(3, 3, CV_32F);
    float theta = float(i) / 12 * CV_PI;
    if(0 <= i && i < 3) // [0, 45)
    {
        topK.at<float>(2, 2) = tan(theta);
        bottomK.at<float>(0, 0) = tan(theta);
        topK.at<float>(1, 2) = 1 - tan(theta);
        bottomK.at<float>(1, 0) = 1 - tan(theta);
    }
    else if(3 <= i && i < 6) // [45, 90)
    {
        theta = CV_PI / 2 - theta;
        topK.at<float>(2, 2) = tan(theta);
        bottomK.at<float>(0, 0) = tan(theta);
        topK.at<float>(2, 1) = 1 - tan(theta);
        bottomK.at<float>(0, 1) = 1 - tan(theta);
    }
    else if(6 <= i && i < 9) // [90, 135)
    {
        theta = theta - CV_PI / 2;
        topK.at<float>(2, 0) = tan(theta);
        bottomK.at<float>(0, 2) = tan(theta);
        topK.at<float>(2, 1) = 1 - tan(theta);
        bottomK.at<float>(0, 1) = 1 - tan(theta);
    }
    else // [135, 180)
    {
        theta = CV_PI - theta;
        topK.at<float>(2, 0) = tan(theta);
        bottomK.at<float>(0, 2) = tan(theta);
        topK.at<float>(1, 0) = 1 - tan(theta);
        bottomK.at<float>(1, 2) = 1 - tan(theta);
    }
}

cv::Mat visuKernel(const cv::Mat &kernel)
{
    cv::Mat result = cv::Mat::zeros(kernel.size(), CV_32FC3);
    float t;
    for(int i = 0; i < kernel.rows; i++)
    {
        for(int j = 0; j < kernel.cols; j++)
        {
            t = kernel.at<float>(i, j);
            if(t > 0)
                result.at<cv::Vec3f>(i, j)[2] = t;
            else if(t < 0)
                result.at<cv::Vec3f>(i, j)[1] = -t;
        }
    }
    return result;
}
cv::Mat mergeTBsideKernels(const cv::Mat &topK, const cv::Mat &bottomK)
{
    cv::Mat result = cv::Mat::zeros(topK.size(), CV_32FC3);
    for(int i = 0; i < topK.rows; i++)
    {
        for(int j = 0; j < topK.cols; j++)
        {
            result.at<cv::Vec3f>(i, j)[2] = topK.at<float>(i,j);
            result.at<cv::Vec3f>(i, j)[1] = bottomK.at<float>(i,j);
        }
    }
    cv::Mat resizedResult;
    cv::resize(result, resizedResult, cv::Size(), 16, 16);
    return resizedResult;
}

EdgeDetectorHyperparams::EdgeDetectorHyperparams()
{
    float theta;
    // Get guassianGradientKernels
    for(int i = 0; i < this->numOrient; i++)
    {
        theta = float(i) / this->numOrient * CV_PI;
        cv::Mat kernel = getGaussianGradientKernel(this->sigmaGG, this->seta, theta);
        kernel = getKernelWithRadiusConstraint(kernel, this->GGKernelRadius, false);
        kernel = discardTinyWeight(kernel, 0.5);
        kernel = equalKernelWeigth(kernel);
        this->GaussianGradientKernels.push_back(kernel);
    }

    // Get avgInhiKernel
    this->avgInhiKernel = getAvgKernel(this->avgKernelSize);
    int avgKCentre = this->avgInhiKernel.rows / 2;
    this->avgInhiKernel.at<float>(avgKCentre, avgKCentre) = 0;
    this->avgInhiKernel = this->avgInhiKernel / cv::sum(this->avgInhiKernel)[0];

    // Get orientKernels, compared to GaborKernel, the result is worse
    //float sigma_x = 0.5;
    //float sigma_y = 2.0;
    //int orientKernelRadius = 3;
    //for(int i = 0; i < this->numOrient_estiOri; i++)
    //{
    //    theta = float(i) / this->numOrient_estiOri * CV_PI + CV_PI / 2;
    //    cv::Mat kernel = getEllipseGaussianKernel(sigma_x, sigma_y, theta);
    //    kernel = getKernelWithRadiusConstraint(kernel, orientKernelRadius, true);
    //   	int center = kernel.rows / 2;
    //    kernel.at<float>(center, center) += 1.;
    //    this->orientKernels.push_back(kernel);
    //}
    //
    //cv::Mat tMergeResult;
    //cv::merge(this->orientKernels, tMergeResult);
    //cv::Mat tMaxResponse;
    //tMaxResponse = findMaxOfChannels(tMergeResult);
    //cv::Mat temp = tMaxResponse - orientKernels[0];
    //float ratio = cv::sum(temp)[0];
    //for(int i = 0; i < this->numOrient_estiOri; i++)
    //{
    //	cv::Mat l = this->orientKernels[i] - tMaxResponse;
    //	l = l.mul(1 / ratio);
    //    this->orientKernels[i] += l;
    //}
    
    // Get orientKernels
    for(int i = 0; i < this->numOrient_estiOri; i++)
    {
        theta = float(i) / this->numOrient_estiOri * CV_PI;
        cv::Mat kernel = getGaborKernel(7, 1.0, theta, 2., 0.5, 0); // getGaborKernel(ksize, sigma, theta, lambda, gamma, psi)
        kernel = equalKernelWeigth(kernel);
        int center = kernel.rows / 2;
        kernel.at<float>(center, center) += 1.;
        this->orientKernels.push_back(kernel);
    }
    // Get topSideKenels bottomSideKernels
    for(int i = 0; i < this->numOrient_estiOri; i++)
    {
        cv::Mat topK, bottomK;
        getTBsideKernels(i, topK, bottomK);
        this->topSideKernels.push_back(topK);
        this->bottomSideKernels.push_back(bottomK);
    }

#if TEST_EDGEDETECTORHYPERPARAMS
    for(int i = 0; i < this->GaussianGradientKernels.size(); i++)
    {
        int delta_angle = 180 / numOrient;
        int thetaV = i * delta_angle;
        writeImage(visuKernel(this->GaussianGradientKernels[i]), "GaussianGradientKernel_" + to_string(i), true);
        cout << this->GaussianGradientKernels[i] << endl;
        cout << "sum: " << cv::sum(GaussianGradientKernels[i])[0] << endl;
    }

    for(int i = 0; i < this->numOrient_estiOri; i++)
    {
        int delta_angle = 180 / numOrient_estiOri;
        int thetaV = i * delta_angle;
        writeImage(visuKernel(this->orientKernels[i]), "OrientKernels_" + to_string(i), true);
    }

    for(int i = 0; i< this->numOrient_estiOri; i++)
    {
        int delta_angle = 180 / numOrient_estiOri;
        cv::Mat t = mergeTBsideKernels(this->topSideKernels[i], this->bottomSideKernels[i]);
        writeImage(t, "TB_" + to_string(i * delta_angle), false);
    }
#endif
}
