#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

template <typename T> class Image : public Mat {
public:
	// Constructors
	Image() {}
	Image(const Mat& A) :Mat(A) {}
	Image(int w, int h, int type) :Mat(h, w, type) {}
	// Accessors
	inline T operator()(int x, int y) const { return at<T>(y, x); }
	inline T& operator()(int x, int y) { return at<T>(y, x); }
	inline T operator()(const Point& p) const { return at<T>(p.y, p.x); }
	inline T& operator()(const Point& p) { return at<T>(p.y, p.x); }
	//
	inline int width() const { return cols; }
	inline int height() const { return rows; }
	// To display a floating type image
	Image<uchar> greyImage() const {
		double minVal, maxVal;
		minMaxLoc(*this, &minVal, &maxVal);
		Image<uchar> g;
		convertTo(g, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
		return g;
	}
};

float normL2(const Mat& img);
float phi(float x);
Mat& Dx(const Mat& img);
Mat& Dy(const Mat& img);
Mat& gradient(const Mat& img);
Mat& vecMul(const Mat& A, const Mat& v);
double NCC(const Image<float>& I1, Point m1, const Image<float>& I2, Point m2, int n);
bool rectContains(int m, int n, Point2i p);

template<typename T>
void warpImage(const Mat& inImg, Mat& outImg, vector<vector<Point2i>> warpingField)
{
	assert(warpingField.size() > 0);
	assert(warpingField.size() == inImg.rows && warpingField[0].size() == inImg.cols);
	int m = inImg.rows, n = inImg.cols, mat_type = inImg.type();
	outImg = Mat::zeros(m, n, mat_type);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			Point2i p(j - warpingField[i][j].x, i - warpingField[i][j].y);
			if (rectContains(m, n, p))
				outImg.at<T>(i, j) = inImg.at<T>(p);
		}
	}
}
