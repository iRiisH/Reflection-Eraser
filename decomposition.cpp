#include "decomposition.h"

#define LAMBDA2 .1
#define LAMBDA3 3000
#define LAMBDAP 100000.


float func_w1d(const Mat& I_t, const Mat& I_Oh, const Mat& I_Bh, const Mat& W_Ot, const Mat& W_Bt)
{
	assert(I_t.depth() == CV_32F && I_Oh.depth() == CV_32F && I_Bh.depth() == CV_32F
		&& W_Ot.depth() == CV_32F && W_Bt.depth() == CV_32F);
	assert(W_Ot.cols == W_Bt.cols && W_Ot.rows == W_Bt.rows);

	Mat m = I_t - vecMul(W_Ot, I_Oh) - vecMul(W_Bt, I_Bh);
	float result = normL2(m);
	result = phi(result);
	result = 1./ result;
	return result;
}

float func_w2d(const Mat& I_Bh)
{
	Mat& dx = Dx(I_Bh), dy = Dy(I_Bh);
	float nx = normL2(dx), ny = normL2(dy);
	return 1. / phi(nx + ny);
}

float func_w3d(const Mat& I_Oh)
{
	return func_w2d(I_Oh); // w2 and w3 are the exact same function, yet for the code
						  // to be clear we create two separate functions
}

float L(const Mat& I_O, const Mat& I_B)
{
	Mat DI_O = gradient(I_O), DI_B = gradient(I_B);
	float res = 0.;
	int m = I_O.rows, n = I_O.cols;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
			res += DI_O.at<float>(i, j) * DI_B.at<float>(i, j);
	}
	return res;
}

float L(const Mat& I_O, const Mat& I_Oh, const Mat& I_B, const Mat& I_Bh)
{
	Mat DI_O = gradient(I_O), DI_B = gradient(I_B), DI_Oh = gradient(I_Oh), DI_Bh = gradient(I_Bh);
	float res = 0.;
	for (int i = 0; i < DI_O.rows; i++)
	{
		for (int j = 0; j < DI_O.cols; j++)
		{
			res += DI_Oh.at<float>(i,j) * DI_B.at<float>(i,j);
			res += DI_O.at<float>(i,j) * DI_Bh.at<float>(i,j);
			res -= DI_Oh.at<float>(i,j) * DI_Bh.at<float>(i,j);
		}
	}
	return res;
}

float objective (const Mat& I_O, const Mat& I_B, const vector<vector<vector<Point2i>>> &V_O_list,
	const vector<vector<vector<Point2i>>> &V_B_list, const vector<Mat&> imgs, const Mat& img_ref)
{
	// be careful that the images are all CV_32F (float) format
	assert(I_O.type() == CV_32F);
	float obj = 0.;

	// natural image smoothness :
	// heavy-tailed distribution on the gradient
	float t_smooth = gradient_normL1(I_O) + gradient_normL1(I_B);
	t_smooth *= LAMBDA2;
	obj += t_smooth;

	// gradient ownership term
	float t_ownership = LAMBDA3 * L(I_O, I_B);
	obj += t_ownership;

	// negativity penalty
	int m = I_O.rows, n = I_O.cols;
	float t_negativity = pow(min(I_B), 2.);
	t_negativity += pow(min(I_O), 2.);
	t_negativity += pow(min(imgMinus (Mat::ones (m, n, CV_32F), I_B)), 2.);
	t_negativity += pow(min(imgMinus(Mat::ones(m, n, CV_32F), I_O)), 2.);
	obj += t_negativity;

	// objective term
	Mat temp = imgMinus(img_ref, I_O);
	float t_main = normL1(imgMinus(temp, I_B));
	for (int i = 0; i < N_IMGS; i++)
	{
		vector<vector<Point2i>> V_O = V_O_list[i], V_B = V_B_list[i];
		Mat I_O_mod = warpedImage(I_O, V_O), I_B_mod = warpedImage(I_B, V_B);
		Mat temp2 = imgMinus(imgs[i], I_O_mod);
		t_main += normL1(imgMinus(temp2, I_B_mod));
	}
	obj += t_main;

	return obj;
}

Mat& imgToVec(const Mat& img)
{
	int m = img.rows, n = img.cols;
	Mat res = Mat::zeros(m*n, 1, CV_64F);
	int m = img.rows, n = img.cols;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			res.at<double>(i*m+j, 1)= (double)img.at<float>(i, j);
		}
	}
	return res;
}

Mat& vecToImg(const double* vec, const int m, const int n)
{
	Mat img = Mat::zeros(m, n, CV_32F);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			img.at<float>(i, j) = (float)vec[m*i + j];
		}
	}
//	delete vec;
	return img;
}

Mat& vecToImg2(const Mat& vec, int m, int n)
{
	Mat img = Mat::zeros(m, n, CV_32F);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			img.at<float>(i, j) = (float)vec.at<double>(i*m+j, 1);
		}
	}
	return img;
}

class Objective_O : public MinProblemSolver::Function
{
private:
	const Mat I_B, img_ref;
	const vector<Mat&> imgs;
	const vector<vector<vector<Point2i>>>& V_O_list, V_B_list;
public:
	Objective_O(Mat& I_Bc, vector<vector<vector<Point2i>>>& V_O_listc,
		vector<vector<vector<Point2i>>>& V_B_listc, vector<Mat&> imgsc, Mat& img_refc):
		I_B(I_Bc), V_O_list (V_O_listc), V_B_list (V_B_listc), imgs(imgsc), img_ref (img_refc)
	{	}
	int getDims() const
	{
		return I_B.rows * I_B.cols;
	}
	double calc(const double* x)const
	{
		Mat I_O = vecToImg(x, I_B.rows, I_B.cols);
		float obj = objective(I_O, I_B, V_O_list, V_B_list, imgs, img_ref);
		return (double)obj;
	}
};

Mat& solve_O (Mat& I_B, Mat& I_O, vector<vector<vector<Point2i>>>& V_O_list,
	vector<vector<vector<Point2i>>>& V_B_list, vector<Mat&> imgs, Mat& img_ref)
{
	cv::Ptr<cv::DownhillSolver> solver = cv::DownhillSolver::create();
	cv::Ptr<cv::MinProblemSolver::Function> ptr_F = cv::makePtr<Objective_O>(I_B, V_O_list,
		V_B_list, imgs, img_ref);
	solver->setFunction(ptr_F);
	Mat x = imgToVec(I_O);
	double res = solver->minimize(x);
	int m = I_O.rows, n = I_O.cols;
	Mat new_I_O = vecToImg2(x, m, n);
	return new_I_O;
}


class Objective_B : public MinProblemSolver::Function
{
private:
	const Mat I_O, img_ref;
	const vector<Mat&> imgs;
	const vector<vector<vector<Point2i>>>& V_O_list, V_B_list;
public:
	Objective_B(Mat& I_Oc, vector<vector<vector<Point2i>>>& V_O_listc,
		vector<vector<vector<Point2i>>>& V_B_listc, vector<Mat&> imgsc, Mat& img_refc) :
		I_O(I_Oc), V_O_list(V_O_listc), V_B_list(V_B_listc), imgs(imgsc), img_ref(img_refc)
	{	}
	int getDims() const
	{
		return I_O.rows * I_O.cols;
	}
	double calc(const double* x)const
	{
		Mat I_B = vecToImg(x, I_O.rows, I_O.cols);
		float obj = objective(I_O, I_B, V_O_list, V_B_list, imgs, img_ref);
		return (double)obj;
	}
};

Mat& solve_B(Mat& I_B, Mat& I_O, vector<vector<vector<Point2i>>>& V_O_list,
	vector<vector<vector<Point2i>>>& V_B_list, vector<Mat&> imgs, Mat& img_ref)
{
	cv::Ptr<cv::DownhillSolver> solver = cv::DownhillSolver::create();
	cv::Ptr<cv::MinProblemSolver::Function> ptr_F = cv::makePtr<Objective_B>(I_O, V_O_list,
		V_B_list, imgs, img_ref);
	solver->setFunction(ptr_F);
	Mat x = imgToVec(I_B);
	double res = solver->minimize(x);
	int m = I_B.rows, n = I_B.cols;
	Mat new_I_B = vecToImg2(x, m, n);
	return new_I_B;
}

void decompose(Mat& I_O, Mat& I_B, vector<vector<vector<Point2i>>>& V_O_list,
	vector<vector<vector<Point2i>>>& V_B_list, vector<Mat&> imgs, Mat& img_ref)
{
	Mat I_O_channels[3], I_B_channels[3], img_ref_channels[3];
	vector<Mat&> imgs_channels[3];
	int m = I_O.rows, n = I_O.cols;
	for (int k = 0; k < 3; k++)
	{
		I_O_channels[k] = Mat::zeros(m, n, CV_32F);
		I_B_channels[k] = Mat::zeros(m, n, CV_32F);
		img_ref_channels[k] = Mat::zeros(m, n, CV_32F);
		for (int l = 0; l < N_IMGS; l++)
		{
			imgs_channels[k][l] = Mat::zeros(m, n, CV_32F);
		}
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				I_O_channels[k].at<float>(i, j) = ((float)I_O.at<Vec3b>(i, j)[k]) / 255.;
				I_B_channels[k].at<float>(i, j) = ((float)I_B.at<Vec3b>(i, j)[k]) / 255.;
				img_ref_channels[k].at<float>(i, j) = ((float)img_ref.at<Vec3b>(i, j)[k]) / 255.;
				for (int l = 0; l < N_IMGS; l++)
					imgs_channels[k][l].at<float>(i, j) = ((float)imgs[l].at<Vec3b>(i, j)[k]) / 255.;
			}
		}
	}
	Mat new_I_O_channels[3], new_I_B_channels[3];
	for (int k = 0; k < 3; k++)
	{
		new_I_O_channels[k] = solve_O(I_B_channels[k], I_O_channels[k], V_O_list, V_B_list, imgs_channels[k], img_ref_channels[k]);
		new_I_B_channels[k] = solve_B(I_B_channels[k], I_O_channels[k], V_O_list, V_B_list, imgs_channels[k], img_ref_channels[k]);
	}
	for (int k = 0; k < 3; k++)
	{
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				I_O.at<Vec3b>(i, j)[k] = (int)((I_O_channels[k]).at<float>(i, j)[k] *255.);
				I_B.at<Vec3b>(i, j)[k] = (int)((I_B_channels[k]).at<float>(i, j)[k] * 255.);
			}
		}
	}
}
