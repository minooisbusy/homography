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
void showResult(Mat, Mat, std::string);
void compute();
Mat homography(std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>);
std::tuple<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> sampling(std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>, std::vector<cv::DMatch>, int n_sample=4);
std::tuple<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>, Mat, Mat> conditioning(std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>);
bool test_collinear(std::vector<cv::KeyPoint>kpts1, float  eps=5.5f);
cv::Point3f point2line(Point2f,Point2f);
float dist(Point2f, Point3f);
std::tuple<unsigned int, std::vector<int>> concensus(std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2, std::vector<cv::DMatch> matches, Mat H, Mat invH, float eps=0.5f);
Point2f Transformation(Mat, Point2f);
float pointNorm(Point2f);
std::tuple<Mat, float, std::vector<DMatch>> ransac(std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2, std::vector<cv::DMatch> matches, float min=0.5f, float p=0.99, float s=4,float eps=0.5f);
bool test_homography(Mat H);
float norm_matrix(Mat a);
};
}

#endif
