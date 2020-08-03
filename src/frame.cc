#include"frame.h"

namespace FRAME
{
FramePair::FramePair(std::string arg1, std::string arg2)
{
    this->src1 = imread(arg1, cv::IMREAD_GRAYSCALE);
    this->src2 = imread(arg2, cv::IMREAD_GRAYSCALE);
    assert(src1.data != NULL);
    assert(src1.data != NULL);
    resize(src1, src1, Size(640, 480));
    resize(src2, src2, Size(640, 480));
}

void FramePair::showPairs()
{
    Mat showImg;
    hconcat(src1, src2,showImg);
    resize(showImg,showImg, Size(640*2, 480));
    
    imshow("Original scene Show", showImg);
    waitKey(0);
}

std::tuple<Mat, Mat> FramePair::get()
{
    return {src1.clone(), src2.clone()};
}
}
