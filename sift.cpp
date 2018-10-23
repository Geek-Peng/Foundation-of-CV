#include "stdafx.h"
#include <iostream>
using namespace std;
using namespace cv;


int _tmain(int argc, _TCHAR* argv[])

{
	
	cout << CV_VERSION;
	
	Mat src1 = imread("E:\\2.jpg");
	Mat src2 = imread("E:\\22.jpg");
	Mat G_src1 = imread("E:\\2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat G_src2 = imread("E:\\22.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	if (!src1.data)
		cout << "找不到图片";
	imshow("sss1", G_src1);
	imshow("sss2",G_src2);
	//waitKey(10000);

	//Ptr<xfeatures2d::SiftFeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create();//特征点探测器
	Ptr<Feature2D> detector = xfeatures2d::SIFT::create(100);//特征点探测器,这里一定要控制最有特征点的个数，否则会出现过匹配问题
	//cv::xfeatures2d::SiftFeatureDetector	detector; 
	std::vector<KeyPoint>  kp1, kp2; //论文中说关键点以向量的形式给出


	//Mat draw1, draw2;
	//detector->detectAndCompute(G_src1,Mat(),kp1,draw1);
	//detector->detectAndCompute(G_src2, Mat(), kp2, draw2);
	detector->detect(G_src1, kp1);
	detector->detect(G_src2, kp2);


Mat des1, des2; //特征描述子


    //cv::xfeatures2d::SiftDescriptorExtractor extractor; //描述符提取器，老版本这么写，新版本不这么写了，新版本直接在SIFT中封装了compute函数
	//extractor.compute(G_src1,kp1,des1); //
	//extractor.compute(G_src2, kp2, des2);
detector->compute(G_src1, kp1, des1);
detector->compute(G_src2, kp2, des2);
	
	Mat res1, res2;
	int drawmode = DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
	drawKeypoints(src1, kp1, res1, Scalar::all(-1), drawmode);//在内存中画出特征点，scalar是一个存储像素的结构体，all(-1)设置所有像素点的值为-1，代表原色
	drawKeypoints(src2, kp2, res2, Scalar::all(-1), drawmode);
	//drawKeypoints(G_src1, kp1, res1, Scalar::all(-1), drawmode);//在内存中画出特征点，scalar是一个存储像素的结构体，all(-1)设置所有像素点的值为-1，代表原色
	//drawKeypoints(G_src2, kp2, res2, Scalar::all(-1), drawmode);
	cout << "图1描述子大小:" << kp1.size() << endl;
	cout << "图2描述子大小:" << kp2.size() << endl;
	
	CvFont font;
	double hscale = 1;
	double vscale = 1;
	int linewidth = 2;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, hscale, vscale, 0, linewidth); //初始化字体 font
	IplImage *transimg1 = cvCloneImage(& (IplImage)res1 ); //将res1强制转换为&(IplImage)类型
	IplImage *transimg2 = cvCloneImage(&(IplImage)res2);
	char str1[20], str2[20];

	int iii = kp1.size();
	sprintf_s(str1, iii, "%d");
	
	iii = kp2.size();
	sprintf_s(str2, iii, "%d");
	


	cvPutText(transimg1,str1,cvPoint(280,230),&font,CV_RGB(255,0,0));//在图片上输出字符
	cvPutText(transimg2, str2, cvPoint(280, 230), &font, CV_RGB(255, 0, 0)); //cvpoint()为起笔的x,y坐标
	cvShowImage("1描述子",transimg1);
	cvShowImage("2描述子", transimg2);
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2,true);
	//BFMatcher matcher(NORM_L2); //老版本这么写
	vector<DMatch> matches;
	//matcher.match(des1,des2,matches);
	matcher->match(des1, des2, matches);
	Mat img_match;
	drawMatches(src1, kp1, src2, kp2, matches,img_match );
	//drawMatches(G_src1, kp1, G_src2, kp2, matches, img_match);
	cout << "匹配的点数：" << matches.size()<< endl;
	imshow("匹配图：", img_match);
	waitKey();
	return 0;
	vector<int > a;

}
