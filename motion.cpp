#include "motion.h"

#define LAMBDA4 0.5

float func_w1m(const Mat& I_t, const Mat& I_O, const Mat& I_B, const vector<vector<Point2i>> V_Oth, const vector<vector<Point2i>> V_Bth, IntVec a)
{
	IntVec pO, pB;
	pO.x = a.x - (V_Oth[a.x][a.y]).x; pO.y = a.y - (V_Oth[a.x][a.y]).y;
	pB.x = a.x - (V_Bth[a.x][a.y]).x; pB.y = a.y - (V_Bth[a.x][a.y]).y;
	float res = I_t.at<float>(a.x, a.y) - I_O.at<float>(pO.x, pO.y) - I_B.at<float>(pB.x, pB.y);
	res = pow(res, 2);
	return 1. / res;	
}

float func_w2m(const Mat& V_Oth)
{
	Mat& grad = gradient(V_Oth);
	float res = normL2(grad);
	return 1. / res;
}

float func_w3m(const Mat& V_Bth)
{
	return func_w2m(V_Bth);
}

Mat& fieldListToVec(const vector<vector<Point2i>>& v)
{
	assert(v.size() > 0);
	assert(v[0].size() > 0);
	int m = v.size(), n = v[0].size();
	Mat res = Mat::zeros(2*m*n, 1, CV_64F);
	
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			res.at<double>(i*m + j, 1) = (double)v[i][j].x;
			res.at<double>(m*n + i*m + j, 1) = (double)v[i][j].y;
		}
	}
	return res;
}

vector<vector<Point2i>>& vecToFieldList(Mat& vec, int m, int n)
{
	vector<vector<Point2i>> res(m);
	
	for (int i = 0; i < m; i++)
		res[i] = vector<Point2i>(n);

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			res[i][j].x = (int)vec.at<double>(i*m + j, 1);
			res[i][j].y = (int)vec.at<double>(m*n + i*m + j, 1);
		}
	}
	
	return res;
}

float objective2(const Mat& I_O, const Mat& I_B, const vector<vector<Point2i>> &V_O,
	const vector<vector<Point2i>> &V_B, const Mat& img)
{
	// be careful that the images are all CV_32F (float) format
	Mat I_O_channels[3], I_B_channels[3], img_channels[3];
	int m = I_O.rows, n = I_O.cols;
	float obj = 0.;
	for (int k = 0; k < 3; k++)
	{
		I_O_channels[k] = Mat::zeros(m, n, CV_32F);
		I_B_channels[k] = Mat::zeros(m, n, CV_32F);
		img_channels[k] = Mat::zeros(m, n, CV_32F);
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				I_O_channels[k].at<float>(i, j) = (float)(I_O.at<Vec3b>(i, j))[k];
				I_B_channels[k].at<float>(i, j) = (float)(I_B.at<Vec3b>(i, j))[k];
				img_channels[k].at<float>(i, j) = (float)(img.at<Vec3b>(i, j))[k];
			}
		}

		// objective term
		Mat I_O_mod = warpedImage(I_O_channels[k], V_O), I_B_mod = warpedImage(I_B_channels[k], V_B);
		Mat temp2 = imgMinus(img_channels[k], I_O_mod);
		float t_main = normL1(imgMinus(temp2, I_B_mod));
		obj += t_main;
	}
	// motion field smoothness term
	float t_smooth = gradient_field_normL1(V_O);
	t_smooth += gradient_field_normL1(V_B);
	t_smooth *= LAMBDA4;
	obj += t_smooth;

	return obj;
}


class Objective_V_O : public MinProblemSolver::Function
{
private:
	const Mat I_O, I_B;
	const Mat& img;
	const vector<vector<Point2i>>& V_B;
public:
	Objective_V_O(Mat& I_Oc, Mat& I_Bc, vector<vector<Point2i>>& V_Bc, Mat& imgc) :
		I_O (I_Oc), I_B (I_Bc), V_B (V_Bc), img (imgc)
	{	}
	int getDims() const
	{
		return I_O.rows * I_O.cols;
	}
	double calc(const double* x)const
	{
		int m = I_O.rows, n = I_O.cols;
		Mat vec = Mat::zeros(m*n, 1, CV_64F);
		for (int i = 0; i < m*n; i++)
			vec.at<double>(i, 1) = x[i];
		vector<vector<Point2i>> V_O = vecToFieldList(vec, m, n);
		float obj = objective2(I_O, I_B, V_O, V_B, img);
		return (double)obj;
	}
};

vector<vector<Point2i>>& solve_V_O(Mat& I_O, Mat& I_B, vector<vector<Point2i>>& V_O,
	vector<vector<Point2i>>& V_B, Mat& img)
{
	cv::Ptr<cv::DownhillSolver> solver = cv::DownhillSolver::create();
	cv::Ptr<cv::MinProblemSolver::Function> ptr_F = cv::makePtr<Objective_V_O>(I_O, I_B, V_B, img);
	solver->setFunction(ptr_F);
	Mat x = fieldListToVec(V_O);
	double res = solver->minimize(x);
	int m = I_O.rows, n = I_O.cols;
	vector<vector<Point2i>> new_V_O = vecToFieldList(x, m, n);
	return new_V_O;
}

class Objective_V_B : public MinProblemSolver::Function
{
private:
	const Mat I_O, I_B;
	const Mat& img;
	const vector<vector<Point2i>>& V_O;
public:
	Objective_V_B(Mat& I_Oc, Mat& I_Bc, vector<vector<Point2i>>& V_Oc, Mat& imgc) :
		I_O(I_Oc), I_B(I_Bc), V_O(V_Oc), img(imgc)
	{	}
	int getDims() const
	{
		return I_O.rows * I_O.cols;
	}
	double calc(const double* x)const
	{
		int m = I_O.rows, n = I_O.cols;
		Mat vec = Mat::zeros(m*n, 1, CV_64F);
		for (int i = 0; i < m*n; i++)
			vec.at<double>(i, 1) = x[i];
		vector<vector<Point2i>> V_B = vecToFieldList(vec, m, n);
		float obj = objective2(I_O, I_B, V_O, V_B, img);
		return (double)obj;
	}
};

vector<vector<Point2i>>& solve_V_B(Mat& I_O, Mat& I_B, vector<vector<Point2i>>& V_O,
	vector<vector<Point2i>>& V_B, Mat& img)
{
	cv::Ptr<cv::DownhillSolver> solver = cv::DownhillSolver::create();
	cv::Ptr<cv::MinProblemSolver::Function> ptr_F = cv::makePtr<Objective_V_B>(I_O, I_B, V_O, img);
	solver->setFunction(ptr_F);
	Mat x = fieldListToVec(V_B);
	double res = solver->minimize(x);
	int m = I_O.rows, n = I_O.cols;
	vector<vector<Point2i>> new_V_B = vecToFieldList(x, m, n);
	return new_V_B;
}

void motionEstimation(Mat& I_O, Mat& I_B, vector<vector<Point2i>>& V_O,
	vector<vector<Point2i>>& V_B, Mat& img)
{
	vector<vector<Point2i>> new_V_O = solve_V_O(I_O, I_B, V_O, V_B, img);
	vector<vector<Point2i>> new_V_B = solve_V_B(I_O, I_B, V_O, V_B, img);
	V_O = new_V_O;
	V_B = new_V_B;
}

void estimateMotion(Mat& I_O, Mat& I_B, vector<vector<vector<Point2i>>>& V_O_list,
	vector<vector<vector<Point2i>>>& V_B_list, vector<Mat&> &imgs)
{
	assert(imgs.size() == N_IMGS);
	for (int k = 0; k < N_IMGS; k++)
		motionEstimation(I_O, I_B, V_O_list[k], V_B_list[k], imgs[k]);
}
