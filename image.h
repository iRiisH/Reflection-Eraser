#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/optim.hpp>

#include <iostream>
#include <fstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N_IMGS 4               // number of images in PATH folder (minus one, that is used as reference frame)
#define PATH "../images/half/" // source folder
#define RESIZE true            // if true, resizes the image with the RESIZE_RATIO
#define RESIZE_RATIO 0.25
#define EDGES_THRESHOLD 20     // the threshold used by the Canny edge detector
#define EDGES_RATIO 3

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
float normL1(const Mat& img);
float normL2(const Mat& img);
float phi(float x);
void Dx(const Mat& img, Mat& deriv);
void Dy(const Mat& img, Mat& deriv);
void gradient(const Mat& img, Mat& grad);
void vecMul(const Mat& A, const Mat& v, Mat& result);
void imgMinus(const Mat& I1, const Mat& I2, Mat& res);
double NCC(const Image<float>& I1, Point m1, const Image<float>& I2, Point m2, int n);
bool rectContains(int m, int n, Point2i p);
float gradient_normL1(const Mat& img);
float gradient_field_normL1(const vector<vector<Point2i>>& v);
float min(const Mat& img);

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

template<typename T>
int min_ind(vector<T> list)
{
	assert(list.size() > 0);
	int n = list.size();
	T val = list[0];
	int ind = 0;
	for (int i = 1; i < n; i++)
	{
		if (list[i] < val)
		{
			val = list[i];
			ind = i;
		}
	}
	return ind;
}

