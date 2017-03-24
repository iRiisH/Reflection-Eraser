#include "motion.h"

float func_w1m(const Mat& I_t, const Mat& I_O, const Mat& I_B, const vector<vector<Point2i>> V_Oth, const vector<vector<Point2i>> V_Bth, IntVec a)
{
	IntVec pO, pB;
	pO.x = a.x - (V_Oth[a.x][a.y]).x; pO.y = a.y - (V_Oth[a.x][a.y]).y;
	pB.x = a.x - (V_Bth[a.x][a.y]).x; pO.y = a.y - (V_Bth[a.x][a.y]).y;
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
	return func_w2(V_Bth);
}
