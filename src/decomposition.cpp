#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#define LAMBDA2 .1
#define LAMBDA3 3000
#define LAMBDA4 .5
#define LAMBDAP 100000.

using namespace cv;
using namespace std;

float func_w1(const Mat& I_t, const Mat& I_Oh, const Mat& I_Bh, const Mat& W_Ot,
	const Mat& W_Ot, const Mat& W_Bt)
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

float func_w2(const Mat& I_Bh)
{
	Mat& dx = Dx(I_Bh), dy = Dy(I_Bh);
	float nx = normL2(dx), dy = normL2(dy);
	return 1. / phi(nx + ny);
}

float func_w3(const Mat& I_Oh)
{
	return func_w2(I_Oh); // w2 and w3 are the exact same function, yet for the code
						  // to be clear we create two separate functions
}

float L(I_O, I_Oh, I_B, I_Bh)
{
	Mat& DI_O = gradient(I_O), DI_B = gradient(I_B), DI_Oh = gradient(I_Oh), DI_Bh = gradient(I_Bh);
	float res = 0.;
	for (int i = 0; i < D_IO.rows; i++)
	{
		for (int j = 0; j < D_IO.cols; j++)
		{
			sum += DI_Oh * DI_B;
			sum += DI_O * DI_Bh;
			sum -= DI_Oh * D_IBh;
		}
	}
	return res;
}
