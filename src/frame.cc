#include"frame.h"

namespace FRAME
{
FramePair::FramePair(std::string arg1, std::string arg2)
{
    this->src1 = imread(arg1, cv::IMREAD_GRAYSCALE);
    this->src2 = imread(arg2, cv::IMREAD_GRAYSCALE);
    assert(src1.data != NULL);
    assert(src1.data != NULL);
    resize(src1, src1, Size(int(src1.rows/4), int(src1.cols/4)));
    resize(src2, src2, Size(int(src2.rows/4), int(src2.cols/4)));

//    resize(src1, src1, Size(480,640)); 
//    resize(src2, src2, Size(480,640));
}

void FramePair::showPairs()
{
    Mat showImg;
    hconcat(src1, src2,showImg);
    
    imshow("Original scene Show", showImg);
    waitKey(0);
}

//inliear theshold!!
//inliear의 인덱스 가져가야함 나중에 인라이어 전체로 호모그래피 다시 구함
void FramePair::showResult(Mat src, Mat tgt,std::string winname)
{
    Mat bgr[3]={Mat::zeros(src.rows,src.cols, CV_8UC1)};
    bgr[1] = src.clone();
    bgr[0] = src.clone();;
    //bgr[0] = Mat::zeros(480,640, CV_8UC1);
    bgr[2] = tgt.clone();
    //imshow("original", bgr[0]);
    //imshow("warpped", bgr[2]);
    Mat mg=Mat::zeros(src.cols,src.rows,CV_8UC3);;
    cv::merge(bgr,3, mg);
    imshow(winname, mg);
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

    std::sort(matches.begin(), matches.end()); // descending ditance

    std::vector<DMatch> good_matches;

    const int ptsPairs = cv::min(1000, (int)(matches.size() *0.20));

    for(int i=0;i<ptsPairs;i++)
    {
        good_matches.push_back(matches[i]);
    }


    Mat model;
    float score;
    std::cout<<"# of good matches= "<<ptsPairs<<std::endl;
    
    // RANSAC
    std::vector<DMatch> inliers;
    // min: minmum concensus distance
    // p = probability for accuracy
    // s = # of samples
    // eps = ratio of in/outlier
    double eps = (matches.size()-good_matches.size())/(double)matches.size();
    std::cout<<"Ratio of In/Outlier = "<<eps<<std::endl;
    std::tie(model,score, inliers) = ransac(kpts1,kpts2,good_matches,1.0f, 0.99, 4.0, eps);
    Mat H8(9, 1, CV_64F, model.ptr<double>());
    std::cout<<"H8=\n"<<model<<std::endl;
    std::cout<<"H8=\n"<<H8<<std::endl;

    // For libraries
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    Mat im_res,im_lib;
    for(int i=0; i<matches.size();i++)
    {
        obj.push_back(kpts1[matches[i].queryIdx].pt);
        scene.push_back(kpts2[matches[i].trainIdx].pt);
    }
    Mat libH = cv::findHomography(obj,scene,RANSAC);
    
    if(score>4)
    { 
    model /= norm_matrix(model);
    cv::SVD svd(model, cv::SVD::NO_UV);
    Mat d = svd.w;
    Mat U = svd.u;
    Mat vt = svd.vt;
    std::cout<<"condition # = "<<d.at<float>(0,0)/d.at<float>(1,0)<<std::endl;
    std::cout<<"singular value = "<<d<<std::endl;
    std::cout<<"score = "<<score<<std::endl;

    //Visualization
    warpPerspective(src1, im_res, model, Size(src1.cols,src1.rows));
    warpPerspective(src1, im_lib, libH, Size(src1.cols,src1.rows));
    
    Mat im_feature;
    cv::drawMatches(this->src1, kpts1, src2, kpts2, inliers, im_feature, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("feature", im_feature);

    showResult(src2, im_lib, "lib");
    showResult(src2, im_res, "skeretch");


    }
    else
    {
        std::cout<<"estimation failed"<<std::endl;
        std::cout<<model<<std::endl;
    }

    
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

void debg(int i)
{
    std::cout<<"debug at "<<i<<std::endl;
}

std::tuple<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> FramePair::sampling(std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2, std::vector<cv::DMatch> matches, int n_sample)
{
    // Make Random quatro indice
    int tmp_idx = -1;
    std::array<int,4> tmp_indice = {-1, -1, -1, -1};
    bool flg_tmp = false; //단일 인덱스 중복도 검사
    int idx_stack=0; // tmp_indice index
    bool flg_comp = false; // model index 중복도 검사
    //results
    std::vector<cv::KeyPoint> key1;
    std::vector<cv::KeyPoint> key2;
    while(true)
    {

        tmp_idx = (double)std::rand()/RAND_MAX*matches.size(); // make random number

        for(int i=0; i<n_sample; i++)
        {
            if(tmp_indice[i] == tmp_idx && tmp_indice[i] != -1)
            {
                flg_tmp = true; // temporary indice is redandunt to tmp_indice
                break; // for loop break;
            }
        }
        
        if(flg_tmp == true)
        {
            flg_tmp = false;
            continue;
        }
        else
        {
            tmp_indice[idx_stack] = tmp_idx;
            idx_stack++;
        }
        
        if(idx_stack>3) // End condition: Stacked four indice.
        {
            // compare stacked indices
            flg_comp = false; // if true, then indice is redundant.
            std::sort(tmp_indice.begin(), tmp_indice.end()); // from zero to ascending?
            // compare with specific indice with tmp_indice
            for(int i=0; i< indice.size();i++)
            {
                std::sort(indice[i].begin(), indice[i].end());
                if(tmp_indice == indice[i])
                {
                    flg_comp = true;
                    break;
                }
            }
            
            
            for(int i=0; i<4; i++)
            {
                int src_idx = matches[tmp_indice[i]].queryIdx;
                int dst_idx = matches[tmp_indice[i]].trainIdx;
                key1.push_back(kpts1[src_idx]);
                key2.push_back(kpts2[dst_idx]);
            }

            if(!(test_collinear(key1)&&test_collinear(key2)) || flg_comp == true)
            {
                //std::cout<<"collinear sample.. resampling.."<<std::endl;
                key1.erase(key1.begin(), key1.end());
                key2.erase(key2.begin(), key2.end());
                tmp_indice = {-1, -1, -1, -1};
                idx_stack = 0;
                flg_comp=false;
                continue;
            }
            if(flg_comp == false)
                break; // while loop break;
        }
    }//While loop end
    indice.push_back(tmp_indice);

    
    return {key1, key2};
}
std::tuple<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>, Mat, Mat> FramePair::conditioning(std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint>kpts2)
{
    Mat T1 = Mat::eye(3,3,CV_32F);
    Mat T2 = Mat::eye(3,3,CV_32F);
    Mat S1 = Mat::eye(3,3,CV_32F);
    Mat S2 = Mat::eye(3,3,CV_32F);

    Point2f cm1(0,0);
    Point2f cm2(0,0);
    float distance1 = 0.0f;
    float distance2 = 0.0f;

    float sx1 = 0.0f;
    float sy1 = 0.0f;
    float sx2 = 0.0f;
    float sy2 = 0.0f;

    // (i) The points are translated so that their centroid is at the origin
    for(int i=0; i<kpts1.size(); i++)
    {
        cm1 +=kpts1[i].pt;
        cm2 +=kpts2[i].pt;
    }
    cm1 /=(float)kpts1.size();
    cm2 /=(float)kpts2.size();
    T1.at<float>(0,2)=-cm1.x;
    T1.at<float>(1,2)=-cm1.y;

    T2.at<float>(0,2)=-cm2.x;
    T2.at<float>(1,2)=-cm2.y;

    // (ii) average distance is equal to sqrt(2)
    for(int i=0; i<kpts1.size();i++)
    {
        kpts1[i].pt.x = kpts1[i].pt.x - cm1.x;
        kpts1[i].pt.y = kpts1[i].pt.y - cm1.y;
        kpts2[i].pt.x = kpts2[i].pt.x - cm2.x;
        kpts2[i].pt.y = kpts2[i].pt.y - cm2.y;
        sx1 += abs(kpts1[i].pt.x);
        sx2 += abs(kpts2[i].pt.x);
        sy1 += abs(kpts1[i].pt.y);
        sy2 += abs(kpts2[i].pt.y);
        //distance1 += cv::sqrt(cv::pow(kpts1[i].pt.x,2.0) + cv::pow(kpts1[i].pt.y,2.0));
        //distance2 += cv::sqrt(cv::pow(kpts2[i].pt.x,2.0) + cv::pow(kpts2[i].pt.y,2.0));
    }

//    distance1 /= float(kpts1.size());
//    distance1 *=cv::sqrt(2); 
//    distance2 /= float(kpts2.size());
//    distance2 *=cv::sqrt(2); 

    sx1 /= float(kpts1.size());
    sx2 /= float(kpts2.size());
    sy1 /= float(kpts1.size());
    sy2 /= float(kpts2.size());
    for(int i=0; i<kpts1.size();i++)
    {
        kpts1[i].pt.x /= sx1;
        kpts1[i].pt.y /= sy1;
        kpts2[i].pt.x /= sx2;
        kpts2[i].pt.y /= sy2;
//        kpts1[i].pt.x /= distance1;
//        kpts2[i].pt /= distance2;
    }
    S1.at<float>(0,0) = cv::sqrt(2.0f)/sx1;
    S1.at<float>(1,1) = cv::sqrt(2.0f)/sy1;

    S2.at<float>(0,0) = cv::sqrt(2.0f)/sx2;
    S2.at<float>(1,1) = cv::sqrt(2.0f)/sy2;

//  std::cout<<"T1="<<std::endl;
//  std::cout<<T1<<std::endl;
    T1 = S1*T1;
    T2 = S2*T2;


    return {kpts1, kpts2, T1, T2};

    
}

//If return is true then this set is non-collinear points.
bool FramePair::test_collinear(std::vector<cv::KeyPoint> kpts1, float  eps)
{
    assert(kpts1.size() > 2); // Two points for joining, and the other is test incidence.
    assert(kpts2.size() > 2);
    //std::cout<<"size of point set="<<kpts1.size()<<std::endl;
    std::list<int>::iterator iter;
    for(int i=0; i<3; i++)
    {
        std::list<int> lt;
        for(int i=0; i<kpts1.size();i++)
            lt.push_back(i);
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
                if(d<eps)
                {
//                    Mat test = Mat::zeros(480,640, CV_8UC3);
//                    for(int i=0; i<kpts1.size();i++)
//                    {
                        //std::cout<<kpts1[i].pt<<std::endl;
//                        cv::circle(test, kpts1[i].pt, 3, cv::Scalar(0,255,0),-1);
                        //std::cout<<kpts2[i].pt<<std::endl;
//                    }
//                    std::cout<<"Collinear!, distance = "<<d<<std::endl;
//                    imshow("test",test);
//                    waitKey(0);
                    return false;
                }
                else
                {
//                    Mat test = Mat::zeros(480,640, CV_8UC3);
//                    std::cout<<"Not collinear!"<<std::endl;
//                    for(int i=0; i<kpts1.size();i++)
                    {
//                        cv::circle(test, kpts1[i].pt, 3, cv::Scalar(0,255,0),-1);
//                        std::cout<<kpts1[i].pt<<std::endl;
                        //std::cout<<kpts2[i].pt<<std::endl;
                    }
//                    imshow("test",test);
//                    waitKey(0);
                    return true;
                }
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
    float u = x.x;
    float v = x.y;
    float a = l.x;
    float b = l.y;
    float c = l.z;
    float nom = cv::abs(a*u+b*v+c);
    float den = cv::sqrt(a*a+b*b);

    return nom/den;
}

std::tuple<unsigned int, std::vector<int>> FramePair::concensus(std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2, std::vector<cv::DMatch> matches, Mat H, Mat invH, float eps)
{
    unsigned int confidence=0;
    std::vector<int> indice_inlier;

    for(int i=0; i<matches.size();i++)
    {
        Point2f src =kpts1[matches[i].queryIdx].pt;
        Point2f tgt =kpts2[matches[i].trainIdx].pt;

        Point2f est = Transformation(H, src);
        Point2f est_ = Transformation(invH, tgt);

        float conf = pointNorm(tgt-est);
        float conf_ = pointNorm(src-est_);
        if(conf<eps)
	{
	    confidence+=1;
            indice_inlier.push_back(i);
	}
    }

    return {confidence, indice_inlier};
}

Point2f FramePair::Transformation(Mat H, Point2f u)
{
    Point2f v;
    float* data = (float*)H.data;
    float den = data[6]*u.x + data[7]*u.y+data[8];
    v.x = (data[0]*u.x + data[1]*u.y + data[2])/den;
    v.y = (data[3]*u.x + data[4]*u.y + data[5])/den;

    return v;
}
float FramePair::pointNorm(Point2f x)
{
    return cv::sqrt(x.x*x.x+x.y*x.y);

}
//TODO: Add output: Inlier Mask to use Non-linear Optimization(LM method)
std::tuple<Mat, float, std::vector<DMatch>> FramePair::ransac(std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2, std::vector<cv::DMatch> matches, float min, float p, float s,float eps)
{
    Mat model;
    int N = RANSACUpdateNumIters(p, eps, s, 1000);
    unsigned int confidence = 0;
    unsigned int temp_conf = 0;
    std::vector<cv::KeyPoint> src_sample, dst_sample;
    Mat T1, T2;
    Mat invT2 = Mat::zeros(3,3,CV_32F);
    std::vector<DMatch> inlier_stack;

    std::vector<int> indice_inlier, temp_indice_inlier;
    std::cout<<"N = "<<N<<std::endl;
    for(int i=0; i<N; i++)
    {
        temp_conf = 0;
        std::tie(src_sample, dst_sample) = sampling(kpts1,kpts2,matches, 4);

        std::tie(src_sample, dst_sample, T1, T2) = conditioning(src_sample, dst_sample);

        Mat H = homography(src_sample, dst_sample);
        Mat invH = homography(dst_sample, src_sample);

        // De-normalize
        H = T2.inv()*H;
        H = H*T1;
        invH = T1.inv()*H*T2;
//        if(!test_homography(H))
//        {
//            --i;
//            continue;
//        }
        //TODO: Get inliers As Mask
        std::tie(temp_conf, temp_indice_inlier) = concensus(kpts1,kpts2, matches, H,invH, min);

        if(temp_conf>confidence)
        {
            std::cout<<"model update! score = "<<confidence<<std::endl;
            confidence = temp_conf;
            indice_inlier = temp_indice_inlier;
            model = H;
        }
    }
    src_sample.erase(src_sample.begin(), src_sample.end());
    dst_sample.erase(dst_sample.begin(), dst_sample.end());
    std::cout<<"# of inliears = "<<indice_inlier.size()<<std::endl;

    // Over-determined term solution ( Regression )
    for(int i=0; i<indice_inlier.size();i++)
    {
	int src_idx = matches[indice_inlier[i]].queryIdx;
	int dst_idx = matches[indice_inlier[i]].trainIdx;
	src_sample.push_back(kpts1[src_idx]);
	dst_sample.push_back(kpts2[dst_idx]);
    inlier_stack.push_back(matches[indice_inlier[i]]);
    }

    std::tie(src_sample, dst_sample, T1, T2) = conditioning(src_sample, dst_sample);

    Mat H = homography(src_sample, dst_sample);

    // De-normalize
    H = T2.inv()*H;
    model = H*T1;


    //TODO: Make inliears list. It could be Samples mask 0 for outlier , 1 for inlier.
    std::cout<<"model in ransac=\n"<<model<<std::endl;
    model.convertTo(model, model.type(), 1./model.at<float>(2,2));
    std::cout<<"model in ransac=\n"<<model<<std::endl;

    return {model, confidence,inlier_stack};
    





    // model -> score (function)
    //
    // max score model store
    // iteration
}
bool FramePair::test_homography(Mat H)
{
    
    H /= H.at<float>(2,2);
    cv::SVD svd(H, cv::SVD::FULL_UV);
    Mat d = svd.w;
    if(d.at<float>(0,0)/d.at<float>(2,2) >100)
        return false;
    std::cout<<"condition number = "<<d.at<float>(0,0)/d.at<float>(2,2)<<std::endl;
    float* data = (float*)H.data;

    float D = data[0]*data[4]-data[1]*data[3];
    float sx = cv::sqrt(data[0]*data[0]+data[3]*data[3]);
    float sy = cv::sqrt(data[1]*data[1]+data[5]*data[5]);
    float P = cv::sqrt(data[6]*data[6]+data[7]*data[7]);
    if(D<=0 || sx>4 || sy<0.1f || sy>4 || P>0.002)
        return false;
    else 
        return true;

}
int FramePair::RANSACUpdateNumIters(double p, double ep, int modelPoints, int maxIters)
{
    if( modelPoints <= 0 )
        CV_Error( Error::StsOutOfRange, "the number of model points should be positive" );

    p = MAX(p, 0.);
    p = MIN(p, 1.);
    ep = MAX(ep, 0.);
    ep = MIN(ep, 1.);

    // avoid inf's & nan's
    double num = MAX(1. - p, DBL_MIN);
    double denom = 1. - std::pow(1. - ep, modelPoints);
    if( denom < DBL_MIN )
        return 0;

    num = std::log(num);
    denom = std::log(denom);

    return denom >= 0 || -num >= maxIters*(-denom) ? maxIters : cvRound(num/denom);
}
float FramePair::norm_matrix(Mat a)
{
    float* data = (float*)a.data;
    float res=0;
    for(int i=0; i<a.cols*a.rows;i++)
        res+=data[i]*data[i];
    res = cv::sqrt(res);

    return res;
}
}
