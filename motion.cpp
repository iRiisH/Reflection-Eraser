#include "motion.h"

#define LAMBDA4 0.5

/*float func_w1m(const Mat& I_t, const Mat& I_O, const Mat& I_B, const vector<vector<Point2i>> V_Oth, const vector<vector<Point2i>> V_Bth, IntVec a)
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
}*/

void fieldListToVec(const vector<vector<Point2i>>& v, Mat& res)
{
	assert(v.size() > 0);
	assert(v[0].size() > 0);
	int m = v.size(), n = v[0].size();
	res = Mat::zeros(2*m*n, 1, CV_64FC1);
	
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			res.at<double>(i*m + j, 0) = (double)v[i][j].x;
			res.at<double>(m*n + i*m + j, 0) = (double)v[i][j].y;
		}
	}
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
			res[i][j].x = (int)vec.at<double>(i*m + j, 0);
			res[i][j].y = (int)vec.at<double>(m*n + i*m + j, 0);
		}
	}
	
	return res;
}

// once again we have to use these as global variables

Mat I_O, I_B, img;
vector<vector<Point2i>> V_O, V_B;

float objective2(const Mat& I_O, const Mat& I_B, const vector<vector<Point2i>> &V_O,
	const vector<vector<Point2i>> &V_B, const Mat& img)
{
	cout << "*";
	// be careful that the images are all CV_32F (float) format
	vector<Mat> I_O_channels(3), I_B_channels(3), img_channels(3);
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
		Mat I_O_mod;
		warpImage<float>(I_O_channels[k], I_O_mod, V_O);
		Mat I_B_mod;
		warpImage<float>(I_B_channels[k], I_B_mod, V_B);
		Mat temp2;
		imgMinus(img_channels[k], I_O_mod, temp2);
		imgMinus(temp2, I_B_mod, temp2);
		float t_main = normL1(temp2);
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

public:
	int getDims() const
	{
		return 2*I_O.rows * I_O.cols;
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

vector<vector<Point2i>>& solve_V_O()
{
	int m = I_O.rows, n = I_O.cols;
	cv::Ptr<cv::DownhillSolver> solver = cv::DownhillSolver::create();

	cv::Ptr<cv::MinProblemSolver::Function> ptr_F = cv::makePtr<Objective_V_O>();
	solver->setFunction(ptr_F);
	Mat initStep = Mat::zeros(2*m*n, 1, CV_64FC1);
	double val = 3.;
	for (int i = 0; i < 2*m*n; i++)
		initStep.at<double>(i, 0) = val;
	solver->setInitStep(initStep);

	Mat x;
	fieldListToVec(V_O, x);

	double res = solver->minimize(x);
	vector<vector<Point2i>> new_V_O = vecToFieldList(x, m, n);
	return new_V_O;
}

class Objective_V_B : public MinProblemSolver::Function
{
public:
	int getDims() const
	{
		return 2*I_O.rows * I_O.cols;
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

vector<vector<Point2i>>& solve_V_B()
{
	int m = I_O.rows, n = I_O.cols;
	cv::Ptr<cv::DownhillSolver> solver = cv::DownhillSolver::create();
	cv::Ptr<cv::MinProblemSolver::Function> ptr_F = cv::makePtr<Objective_V_B>();
	solver->setFunction(ptr_F);
	Mat initStep = Mat::zeros(m*n, 1, CV_64FC1);
	double val = 3.;
	for (int i = 0; i < 2*m*n; i++)
		initStep.at<double>(i, 0) = val;
	solver->setInitStep(initStep);
	Mat x;
	fieldListToVec(V_B, x);
	double res = solver->minimize(x);
	vector<vector<Point2i>> new_V_B = vecToFieldList(x, m, n);
	return new_V_B;
}

void motionEstimation()
{
	vector<vector<Point2i>> new_V_O = solve_V_O();
	vector<vector<Point2i>> new_V_B = solve_V_B();
	V_O = new_V_O;
	V_B = new_V_B;
}

void estimateMotion(Mat& I_Oc, Mat& I_Bc, vector<vector<vector<Point2i>>>& V_O_list,
	vector<vector<vector<Point2i>>>& V_B_list, vector<Mat> &imgs)
{
	cout << "Motion estimation" << endl;
	assert(imgs.size() == N_IMGS);
	I_O = I_Oc;
	I_B = I_Bc;
	for (int k = 0; k < N_IMGS; k++)
	{
		V_O = V_O_list[k];
		V_B = V_B_list[k];
		img = imgs[k];
		motionEstimation();
	}
}
