#include "decomposition.h"

#define LAMBDA2 .1
#define LAMBDA3 3000
#define LAMBDAP 100000.


/*float func_w1d(const Mat& I_t, const Mat& I_Oh, const Mat& I_Bh, const Mat& W_Ot, const Mat& W_Bt)
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
}*/

float L(const Mat& I_O, const Mat& I_B)
{
	Mat DI_O, DI_B; 
	gradient(I_O, DI_O);
	gradient(I_B, DI_B);
	float res = 0.;
	int m = I_O.rows, n = I_O.cols;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
			res += DI_O.at<float>(i, j) * DI_B.at<float>(i, j);
	}
	return res;
}

/*float L(const Mat& I_O, const Mat& I_Oh, const Mat& I_B, const Mat& I_Bh)
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
}*/

// in order not to copy these images, we use the images as global variables,
// as we else encounter insufficient memory problems

vector<Mat> I_O_channels, I_B_channels;
vector<vector<vector<Point2i>>> V_O_list, V_B_list;
vector<vector<Mat>> imgs_channels;
vector<Mat> img_ref_channels;
Mat temp_I_O, temp_I_B;

float objective1(const Mat& I_O, const Mat& I_B, int k)
{
	// be careful that the images are all CV_32F (float) format
	assert(I_O.type() == CV_32F && I_B.type() == CV_32F &&
		imgs_channels[k][0].type() == CV_32F && img_ref_channels[k].type() == CV_32F);
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
	Mat cont;
	imgMinus(Mat::ones(m, n, CV_32F), I_B, cont);
	t_negativity += pow(min(cont), 2.);
	imgMinus(Mat::ones(m, n, CV_32F), I_O, cont);
	t_negativity += pow(min(cont), 2.);
	obj += t_negativity;
	// objective term
	Mat temp;
	imgMinus(img_ref_channels[k], I_O, temp);
	imgMinus(temp, I_B, temp);
	float t_main = normL1(temp);

	for (int i = 0; i < N_IMGS; i++)
	{
		vector<vector<Point2i>> V_O = V_O_list[i], V_B = V_B_list[i];
		Mat I_O_mod, I_B_mod;
		warpImage<float>(I_O, I_O_mod, V_O);
		warpImage<float>(I_B, I_B_mod, V_B);
		Mat temp2;
		imgMinus(imgs_channels[k][i], I_O_mod, temp2);
		imgMinus(temp2, I_B_mod, temp2);
		t_main += normL1(temp2);
	}
	obj += t_main;
	return obj;
}

Mat& imgToVec(const Mat& img)
{
	int m = img.rows, n = img.cols;
	Mat res = Mat::zeros(m*n, 1, CV_64FC1);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			res.at<double>(i*m+j, 0)= (double)img.at<float>(i, j);
		}
	}
	return res;
}

void vecToImg(const double* vec, const int m, const int n, bool is_I_O)
{
	
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (is_I_O)
				temp_I_O.at<float>(i, j) = (float)vec[m*i + j];
			else
				temp_I_B.at<float>(i, j) = (float)vec[m*i + j];
		}
	}
}

class Objective_O : public MinProblemSolver::Function
{
private:
	const int k;
public:
	Objective_O(int channel) : k(channel)
	{   }
	int getDims() const
	{
		return I_B_channels[k].rows * I_B_channels[k].cols;
	}
	double calc(const double* x)const
	{
		vecToImg(x, I_B_channels[k].rows, I_B_channels[k].cols, true);
		float obj = objective1(temp_I_O, I_B_channels[k], k);
		return (double)obj;
	}
};

void solve_O (int channel)
{
	int m = I_O_channels[0].rows, n = I_O_channels[0].cols;
	cv::Ptr<cv::DownhillSolver> solver = cv::DownhillSolver::create();
	cv::Ptr<cv::MinProblemSolver::Function> ptr_F = cv::makePtr<Objective_O>(channel);
	solver->setFunction(ptr_F);
	Mat initStep = Mat::zeros(m*n, 1, CV_64FC1);
	double val =INIT_SPREAD;
	for (int i = 0; i < m*n; i++)
		initStep.at<double>(i, 0) = val;
	solver->setInitStep(initStep);
	TermCriteria tc(TermCriteria::MAX_ITER + TermCriteria::EPS, MAX_ITERATIONS, CONV_PRECISION);
	solver->setTermCriteria(tc);
	Mat x = imgToVec(I_O_channels[channel]);
	double res = solver->minimize(x);
//	Mat new_I_O = vecToImg2(x, m, n);
//	return new_I_O;
}


class Objective_B : public MinProblemSolver::Function
{
private:
	const int k;
public:
	Objective_B(int channel) : k(channel)
	{   }
	int getDims() const
	{
		return I_O_channels[k].rows * I_O_channels[k].cols;
	}
	double calc(const double* x)const
	{
		vecToImg(x, I_B_channels[k].rows, I_B_channels[k].cols, false);
		float obj = objective1(I_O_channels[k], temp_I_B, k);
		return (double)obj;
	}
};

void solve_B(int channel)
{
	int m = I_O_channels[0].rows, n = I_O_channels[0].cols;
	cv::Ptr<cv::DownhillSolver> solver = cv::DownhillSolver::create();
	cv::Ptr<cv::MinProblemSolver::Function> ptr_F = cv::makePtr<Objective_B>(channel);
	solver->setFunction(ptr_F);
	Mat initStep = Mat::zeros(m*n, 1, CV_64FC1);
	double val = INIT_SPREAD;
	for (int i = 0; i < m*n; i++)
		initStep.at<double>(i, 0) = val;
	solver->setInitStep(initStep);
	TermCriteria tc(TermCriteria::MAX_ITER + TermCriteria::EPS, 500, 0.001);
	solver->setTermCriteria(tc);
	Mat x = imgToVec(I_B_channels[channel]);
	double res = solver->minimize(x);
//	Mat new_I_B = vecToImg2(x, m, n);
//	return new_I_B;
}

void decompose(Mat& I_O, Mat& I_B, vector<vector<vector<Point2i>>>& V_O_listc,
	vector<vector<vector<Point2i>>>& V_B_listc, vector<Mat>& imgs, Mat& img_ref)
{
	cout << "Image decomposition" << endl;
	I_O_channels = vector<Mat>(3);
	I_B_channels = vector<Mat>(3);
	img_ref_channels = vector<Mat>(3);
	imgs_channels = vector<vector<Mat>> (3);
	V_B_list = V_B_listc;
	V_O_list = V_O_listc;
	int m = I_O.rows, n = I_O.cols;
	temp_I_O = Mat::zeros(m, n, CV_32F);
	temp_I_B = Mat::zeros(m, n, CV_32F);
	for (int k = 0; k < 3; k++)
	{
		I_O_channels[k] = Mat::zeros(m, n, CV_32F);
		I_B_channels[k] = Mat::zeros(m, n, CV_32F);
		img_ref_channels[k] = Mat::zeros(m, n, CV_32F);
		imgs_channels[k] = vector<Mat>(N_IMGS);
		for (int l = 0; l < N_IMGS; l++)
			imgs_channels[k][l] = Mat::zeros(m, n, CV_32F);
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
	Mat new_I_O = Mat::zeros (m, n, CV_8UC3), new_I_B = Mat::zeros(m, n, CV_8UC3);
	
	for (int k = 0; k < 3; k++)
	{
		solve_O(k);
		solve_B(k);
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				new_I_O.at<Vec3b>(i, j)[k] = (int)((I_O_channels[k]).at<float>(i, j) *255.);
				new_I_B.at<Vec3b>(i, j)[k] = (int)((I_B_channels[k]).at<float>(i, j) * 255.);
			}
		}
	}
	I_O = new_I_O;
	I_B = new_I_B;
	imwrite("../t1.png", I_O);
	imwrite("../t2.png", I_B);
}
