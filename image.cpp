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

float normL1(const Mat& img)
{
	assert(img.depth() == CV_32F);
	int m = img.rows, n = img.cols;
	float sum = 0.;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			float val = img.at<float>(i, j);
			sum += abs(val);
		}
	}
	return sum;
}

float phi(float x)
{
	return sqrt(x + pow(EPSILON, 2));
}

void Dx(const Mat& img, Mat& deriv)
{
	assert(img.depth() == CV_32F);

	int m = img.rows, n = img.cols;
	deriv = Mat::zeros(m, n, CV_32F);

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n-1; j++)
			deriv.at<float>(i, j) = img.at<float>(i, j+1) - img.at<float>(i, j);
	}
}

void Dy(const Mat& img, Mat& deriv)
{
	assert(img.depth() == CV_32F);

	int m = img.rows, n = img.cols;
	deriv = Mat::zeros(m, n, CV_32F);

	for (int i = 0; i < m-1; i++)
	{
		for (int j = 0; j < n; j++)
			deriv.at<float>(i, j) = img.at<float>(i+1, j) - img.at<float>(i, j);
	}
}

void gradient(const Mat& img, Mat& grad)
{
	assert(img.depth() == CV_32F);
	int m = img.rows, n = img.cols;
	grad = Mat::zeros(m, n, CV_32F);

	for (int i = 0; i < m - 1; i++)
	{
		for (int j = 0; j < n-1; j++)
		{
			float dx = img.at<float>(i, j+1) - img.at<float>(i, j);
			float dy = img.at<float>(i + 1, j) - img.at<float>(i, j);
			grad.at<float>(i, j) = dx*dx + dy*dy;
		}
	}
}

float gradient_normL1(const Mat& img)
{
	int m = img.rows, n = img.cols;
	float res = 0.;
	for (int i = 0; i < m - 1; i++)
	{
		for (int j = 0; j < n - 1; j++)
		{
			float dx = img.at<float>(i, j + 1) - img.at<float>(i, j);
			float dy = img.at<float>(i + 1, j) - img.at<float>(i, j);
			res += abs(dx) + abs(dy);
		}
	}
	return res;
}

float gradient_field_normL1(const vector<vector<Point2i>>& v)
{
	assert(v.size() > 0);
	int m = v.size(), n = v[0].size();
	float res = 0.;
	for (int i = 0; i < m - 1; i++)
	{
		for (int j = 0; j < n - 1; j++)
		{
			float dx = pow ((float)v[i][j + 1].x - v[i][j].x, 2.) + pow ((float)v[i][j+1].y - v[j][j].y, 2.);
			float dy = pow((float)v[i+1][j].x - v[i][j].x, 2.) + pow((float)v[i+1][j].y - v[j][j].y, 2.);
			res += dx + dy;
		}
	}
	return res;
}

void vecMul(const Mat& A, const Mat& v, Mat& res)
{
	assert(A.cols == v.cols * v.rows && A.rows == v.cols * v.rows);

	res = Mat::zeros(v.rows, v.cols, CV_32F);

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
}

void imgMinus(const Mat& I1, const Mat& I2, Mat& res)
{
	assert(I1.type() == CV_32F);
	int m = I1.rows, n = I1.cols;
	res = Mat::zeros(m, n, CV_32F);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
			res.at<float>(i, j) = I1.at<float>(i, j) - I2.at<float>(i, j);
	}
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

bool rectContains(int m, int n, Point2i p)
{
	Rect r(0, 0, n, m);
	return r.contains(p);
}

float min(const Mat& img)
{
	assert(img.type() == CV_32F);
	int m = img.rows, n = img.cols;
	float min = 0.;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			float val = img.at<float>(i, j);
			min = (val < min) ? val : min;
		}
	}
	return min;
}
