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

    std::srand(0);
}

void FramePair::showPairs()
{
    Mat showImg;
    hconcat(src1, src2,showImg);
    
    imshow("Original scene Show", showImg);
    waitKey(0);
}

void FramePair::showResult(Mat src, Mat tgt)
{
    Mat bgr[3]={Mat::zeros(480,640, CV_8UC1)};
    bgr[0] = src.clone();
    bgr[1] = Mat::zeros(480,640, CV_8UC1);
    bgr[2] = tgt.clone();
    Mat mg=Mat::zeros(640,480,CV_8UC3);;
    cv::merge(bgr,3, mg);
    imshow("warped image residual", mg);
    imshow("original", bgr[0]);
    imshow("warpped", bgr[2]);
    waitKey(0);
}

std::tuple<Mat, Mat> FramePair::get()
{
    return {src1.clone(), src2.clone()};
}

void FramePair::compute()
{
    // Compute feature pairs (Library)
    Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
    Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    detector->detectAndCompute(src1,cv::noArray(),kpts1, dscpt1);
    detector->detectAndCompute(src2,cv::noArray(),kpts2, dscpt2);
    matcher->match(dscpt1,dscpt2,matches);


    // RANSAC
    // for loop N=(1-p)/(log(p)~~~ 99%->p=0.99, inlier:outlier=50:50-> s=0.5
        std::vector<cv::KeyPoint> src_sample, dst_sample;
        Mat T1, T2;
        Mat invT2 = Mat::zeros(3,3,CV_32F);
        std::tie(src_sample, dst_sample) = sampling(kpts1,kpts2,matches);

        std::tie(src_sample, dst_sample, T1, T2) = conditioning(src_sample, dst_sample);

        Mat H = homography(src_sample, dst_sample);
        // De-normalize
        std::cout<<"H prime=\n"<<H<<std::endl;
        H = T2.inv()*H;
        H = H*T1;
        H /=H.at<float>(2,2);
        std::cout<<T1<<std::endl;
        std::cout<<T2<<std::endl;
        std::cout<<"H=\n"<<H<<std::endl;

        Mat im_res;
        warpPerspective(src1, im_res, H, Size(640,480));
        
        showResult(src2, im_res);



        // model -> score (function)
        //
        // max score model store
        // iteration
    
    // nonlinear optimization (YET)




    //std::cout<<"-----------"<<std::endl;
    //std::cout<<matches[i].imgIdx<<std::endl; // Don't care 2장 비교시 필요없음 
    //std::cout<<matches[i].queryIdx<<std::endl; // src1 kpt index
    //std::cout<<matches[i].trainIdx<<std::endl; // src2 kpt index
    //std::cout<<matches[i].distance<<std::endl; // matching cost(low value is better)
}

Mat FramePair::homography(std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2)
{
    assert(kpts1.size() == kpts2.size());
    assert(kpts1.size() !=0 );
    //conditioning

    Mat A = Mat::zeros(kpts1.size()*2, 9, CV_32F);
    float *data = (float*)A.data;
    const int cols = A.cols;

    for(long unsigned int i=0; i<kpts1.size(); i++)
    {
        float x = kpts1[i].pt.x;
        float y = kpts1[i].pt.y;

        float x_ = kpts2[i].pt.x;
        float y_ = kpts2[i].pt.y;

        // Odd Row
        data[cols*(i*2) + 3] = -x;
        data[cols*(i*2) + 4] = -y;
        data[cols*(i*2) + 5] = -1.0f;

        data[cols*(i*2) + 6] = y_*x;
        data[cols*(i*2) + 7] = y_*y;
        data[cols*(i*2) + 8] = y_;

        // Even Row
        data[cols*(i*2+1) + 0] = x;
        data[cols*(i*2+1) + 1] = y;
        data[cols*(i*2+1) + 2] = 1.0f;

        data[cols*(i*2+1) + 6] = -x_*x;
        data[cols*(i*2+1) + 7] = -x_*y;
        data[cols*(i*2+1) + 8] = -x_;
    }

    Mat VT;
    cv::SVD svd(A, cv::SVD::FULL_UV);
    Mat H = svd.vt.row(svd.vt.rows-1);
    // H prime
    H = H.reshape(3,3);
    Mat res = Mat::zeros(3,3,CV_32FC1);
    for(int i=0;i<3;i++)
        for(int j=0; j<3;j++)
            res.at<float>(i,j)=H.at<float>(i,j);


    return res;
}

std::tuple<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> FramePair::sampling(std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2, std::vector<cv::DMatch> matches)
{
    // Make Random quatro indice
    int tmp_idx = -1;
    std::array<int,4> tmp_indice = {-1, -1, -1, -1};
    bool flg_tmp = false; //단일 인덱스 중복도 검사
    int idx_stack=0; // tmp_indice index
    bool flg_comp = false; // model index 중복도 검사
    while(true)
    {
        tmp_idx = std::rand()%matches.size();

        for(int i=0; i<4; i++)
        {
            if(tmp_indice[i] == tmp_idx)
            {
                flg_tmp = true;
                break; // for loop break;
            }
        }

        if(flg_tmp == false)
        {
            tmp_indice[idx_stack] = tmp_idx;
            idx_stack++;
        }        
        flg_tmp = false;
        
        if(idx_stack>3)
        {
            // compare stacked indices
            flg_comp = false;
            std::sort(tmp_indice.begin(), tmp_indice.end());
            for(int i=0; i< indice.size();i++)
            {
                std::sort(indice[i].begin(), indice[i].end());
                for(int j=0; j<4;j++)
                {
                    if(tmp_indice[j] != indice[i][j])
                    {
                        flg_comp = true;
                        break;
                    }
                }
                if(flg_comp == true)
                    break;
            }
            if(flg_comp == false)
                break; // while loop break;
        }
    }
    indice.push_back(tmp_indice);

    std::vector<cv::KeyPoint> key1;
    std::vector<cv::KeyPoint> key2;
    for(int i=0; i<4; i++)
    {
        int src_idx = matches[tmp_indice[i]].queryIdx;
        int dst_idx = matches[tmp_indice[i]].trainIdx;
        key1.push_back(kpts1[src_idx]);
        key2.push_back(kpts2[dst_idx]);
    }
    
    return {key1, key2};
}
std::tuple<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>, Mat, Mat> FramePair::conditioning(std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint>kpts2)
{
    Mat T1 = Mat::zeros(3,3,CV_32F);
    Mat T2 = Mat::zeros(3,3,CV_32F);
    Point2f center1(0,0);
    Point2f center2(0,0);
    float distance1 = 0.0f;
    float distance2 = 0.0f;

    // (i) The points are translated so that their centroid is at the origin
    for(int i=0; i<kpts1.size(); i++)
    {
        center1 +=kpts1[i].pt;
        center2 +=kpts2[i].pt;
    }
    center1 /=(float)kpts1.size();
    center2 /=(float)kpts2.size();
    T1.at<float>(0,2)=-center1.x;
    T1.at<float>(1,2)=-center1.y;

    T2.at<float>(0,2)=-center2.x;
    T2.at<float>(1,2)=-center2.y;

    // (ii) average distance is equal to sqrt(2)
    for(int i=0; i<kpts1.size();i++)
    {
        kpts1[i].pt = kpts1[i].pt - center1;
        kpts2[i].pt = kpts2[i].pt - center2;
        distance1 += cv::sqrt(cv::pow(kpts1[i].pt.x,2.0) + cv::pow(kpts1[i].pt.y,2.0));
        distance2 += cv::sqrt(cv::pow(kpts2[i].pt.x,2.0) + cv::pow(kpts2[i].pt.y,2.0));
    }
    distance1 /= float(kpts1.size());
    distance1 *=cv::sqrt(2); 
    distance2 /= float(kpts2.size());
    distance2 *=cv::sqrt(2); 
    for(int i=0; i<kpts1.size();i++)
    {
        kpts1[i].pt /= distance1;
        kpts2[i].pt /= distance2;
    }
    T1.at<float>(0,0) = 1.0f/distance1;
    T1.at<float>(1,1) = 1.0f/distance1;
    T1.at<float>(2,2) = 1.0f;

    T2.at<float>(0,0) = 1.0f/distance2;
    T2.at<float>(1,1) = 1.0f/distance2;
    T2.at<float>(2,2) = 1.0f;

    return {kpts1, kpts2, T1, T2};

    
}
bool FramePair::test_collinear(std::vector<cv::KeyPoint> kpts1, float  eps=0.5f)
{
    assert(kpts1.size() == 4);
    assert(kpts2.size() == 4);
    std::list<int>::iterator iter;
    for(int i=0; i<3; i++)
    {
        std::list lt(0,4);
        for(int j=i+1; j<4; j++)
        {
            Point2f u = kpts1[i].pt;
            Point2f v = kpts1[j].pt;
            Point3f l = point2line(u,v);
            lt.remove(i);
            lt.remove(j);
            for(iter=lt.begin();iter != lt.end(); ++iter)
            {
                float d = dist(kpts1[*iter].pt,l);
                if(d<eps) return false;
            }
        }
    }

    return true;

}
cv::Point3f FramePair::point2line(Point2f u,Point2f v)
{
    float x1 = u.x;
    float y1 = u.y;
    float x2 = v.x;
    float y2 = v.y;

    return Point3f(y1-y2, x2-x1, x1*y2-y1*x2);
}

float FramePair::dist(Point2f x, Point3f l)
{
    float x1 = x.x;
    float y1 = x.y;
    float a = l.x;
    float b = l.y;
    float c = l.z;
    float nom = a*x1+b*y1+c;
    float den = cv::sqrt(a*a+b*b);

    return nom/den;
}
}
