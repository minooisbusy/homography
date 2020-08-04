#ifndef _FRMAE_H_
#define _FRMAE_H_ 
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>
#include<opencv2/features2d.hpp>
#include<string>
#include<tuple>
#include<assert.h>
#include<iostream>
#include<cstdlib>
using namespace cv;
namespace FRAME
{
class FramePair
{
private:
Mat src1;
Mat src2;
std::vector<cv::KeyPoint> kpts1;
std::vector<cv::KeyPoint> kpts2;
Mat dscpt1;
Mat dscpt2;
std::vector<cv::DMatch> matches;

std::vector<std::array<int,4>> indice;

public:
FramePair(std::string, std::string);
std::tuple<Mat, Mat> get();

void showPairs();
void compute();
void ransac(std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>, std::vector<cv::DMatch>);
void homography(std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>);
std::tuple<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> sampling(std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>, std::vector<cv::DMatch>);
std::tuple<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> conditioning(std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>);


};
}

#endif
