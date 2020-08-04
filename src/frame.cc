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
    resize(showImg,showImg, Size(640*2, 480));
    
    imshow("Original scene Show", showImg);
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

    std::vector<cv::KeyPoint> src_sample, dst_sample;
    std::tie(src_sample, dst_sample) = sampling(kpts1,kpts2,matches);

    homography(src_sample, dst_sample);




    //std::cout<<"-----------"<<std::endl;
    //std::cout<<matches[i].imgIdx<<std::endl; // Don't care
    //std::cout<<matches[i].queryIdx<<std::endl; // src1 kpt index
    //std::cout<<matches[i].trainIdx<<std::endl; // src2 kpt index
    //std::cout<<matches[i].distance<<std::endl; // matching cost(low value is better)
}

void FramePair::homography(std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2)
{
    assert(kpts1.size() == kpts2.size());
    assert(kpts1.size() !=0 );

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
    H = H.reshape(3,3);
    std::cout<<H<<std::endl;
}

std::tuple<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> FramePair::sampling(std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2, std::vector<cv::DMatch> matches)
{
    // Make Random quatro indice
    int tmp_idx = -1;
    std::array<int,4> tmp_indice = {-1, -1, -1, -1};
    bool flg_tmp = false;
    int idx_stack=0;
    bool flg_comp = false;
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
std::tuple<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> conditioning(std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint>kpts2)
{
    Mat T1 = Mat::zeros(3,3,CV_32F);
    Mat T2 = Mat::zeros(3,3,CV_32F);
    Point2f center1(0,0);
    Point2f center2(0,0);
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
}
}
