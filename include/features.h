#ifndef _FEATURES_H_
#define _FEATURES_H_
#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<tuple>
using namespace cv;

namespace FEATURES
{
class PairPoint
{
public:
cv::Point2i first;
cv::Point2i second;
float score;
PairPoint(Point2i _first, Point2i _second);
};
class SIFT
{
private:
Mat src1;
Mat src2;
public:
std::vector<PairPoint> getPairs();
};
}

#endif
