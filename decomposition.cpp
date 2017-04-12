#include "decomposition.h"

#define LAMBDA2 .1
#define LAMBDA3 3000
#define LAMBDA4 .5
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

void decompose()
{
}