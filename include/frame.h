#ifndef _FRMAE_H_
#define _FRMAE_H_ 
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<string>
#include<tuple>
#include<assert.h>
using namespace cv;
namespace FRAME
{
class FramePair
{
private:
Mat src1;
Mat src2;

public:
FramePair(std::string, std::string);
std::tuple<Mat, Mat> get();

void showPairs();

};
}

#endif
