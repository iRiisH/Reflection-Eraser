#include "image.h"

#define EPSILON .0001

using namespace cv;
using namespace std;

float normL2(const Mat& img)
{
	assert(img.depth() == CV_32F);
	int m = img.rows, n = img.cols;
	float squared_sum = 0.;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			float val = img.at<float>(i, j);
			squared_sum += pow(val, 2);
		}
	}
	return squared_sum;
}

float phi(float x)
{
	return sqrt(x + pow(EPSILON, 2));
}

Mat& Dx(const Mat& img)
{
	assert(img.depth() == CV_32F);

	int m = img.rows, n = img.cols;
	Mat deriv = Mat::zeros(m, n, CV_32F);

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n-1; j++)
			deriv.at<float>(i, j) = img.at<float>(i, j+1) - img.at<float>(i, j);
	}
	return deriv;
}

Mat& Dy(const Mat& img)
{
	assert(img.depth() == CV_32F);

	int m = img.rows, n = img.cols;
	Mat deriv = Mat::zeros(m, n, CV_32F);

	for (int i = 0; i < m-1; i++)
	{
		for (int j = 0; j < n; j++)
			deriv.at<float>(i, j) = img.at<float>(i+1, j) - img.at<float>(i, j);
	}
	return deriv;
}

Mat& gradient(const Mat& img)
{
	assert(img.depth() == CV_32F);
	int m = img.rows, n = img.cols;
	Mat grad = Mat::zeros(m, n, CV_32F);

	for (int i = 0; i < m - 1; i++)
	{
		for (int j = 0; j < n; j++)
		{
			float dx = img.at<float>(i, j+1) - img.at<float>(i, j);
			float dy = img.at<float>(i + 1, j) - img.at<float>(i, j);
			grad.at<float>(i, j) = dx*dx + dy*dy;
		}
	}
	return grad;
}

Mat& vecMul(const Mat& A, const Mat& v)
{
	assert(A.cols == v.cols * v.rows && A.rows == v.cols * v.rows);

	Mat res = Mat::zeros(v.rows, v.cols, CV_32F);

	for (int i = 0; i < v.rows; i++)
	{
		for (int j = 0; j < v.cols; j++)
		{
			float sum = 0.;
			for (int k = 0; k < A.cols; k++)
				sum += A.at<float>(i + j*v.rows, k);
			res.at<float>(i, j) = sum;
		}
	}
	return res;
}

// Correlation
double mean(const Image<float>& I, Point m, int n) {
	double s = 0;
	for (int j = -n; j <= n; j++)
		for (int i = -n; i <= n; i++)
			s += I(m + Point(i, j));
	return s / (2 * n + 1) / (2 * n + 1);
}

double corr(const Image<float>& I1, Point m1, const Image<float>& I2, Point m2, int n) {
	double M1 = mean(I1, m1, n);
	double M2 = mean(I2, m2, n);
	double rho = 0;
	for (int j = -n; j <= n; j++)
		for (int i = -n; i <= n; i++) {
			rho += (I1(m1 + Point(i, j)) - M1)*(I2(m2 + Point(i, j)) - M2);
		}

	return rho;
}

double NCC(const Image<float>& I1, Point m1, const Image<float>& I2, Point m2, int n) {
	if (m1.x<n || m1.x >= I1.width() - n || m1.y<n || m1.y >= I1.height() - n) return -1;
	if (m2.x<n || m2.x >= I2.width() - n || m2.y<n || m2.y >= I2.height() - n) return -1;
	double c1 = corr(I1, m1, I1, m1, n);
	if (c1 == 0) return -1;
	double c2 = corr(I2, m2, I2, m2, n);
	if (c2 == 0) return -1;
	return corr(I1, m1, I2, m2, n) / sqrt(c1*c2);
}
