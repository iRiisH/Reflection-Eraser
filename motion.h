#pragma once

#include "image.h"

struct IntVec
{
	int x;
	int y;
};

float func_w1m(const Mat& I_t, const Mat& I_O, const Mat& I_B, vector<vector<Point2i>> V_Oth, vector<vector<Point2i>> V_Bth, IntVec a);
float func_w2m(const Mat& V_Oth);
float func_w3m(const Mat& V_Bth);
Mat& fieldListToVec(const vector<vector<Point2i>>& v);
vector<vector<Point2i>>& vecToFieldList(Mat& vec, int m, int n);
float objective2(const Mat& I_O, const Mat& I_B, const vector<vector<Point2i>> &V_O,
	const vector<vector<Point2i>> &V_B, const Mat& img);
vector<vector<Point2i>>& solve_V_O(Mat& I_O, Mat& I_B, vector<vector<Point2i>>& V_O,
	vector<vector<Point2i>>& V_B, Mat& img);
vector<vector<Point2i>>& solve_V_B(Mat& I_O, Mat& I_B, vector<vector<Point2i>>& V_O,
	vector<vector<Point2i>>& V_B, Mat& img);
void motionEstimation(Mat& I_O, Mat& I_B, vector<vector<Point2i>>& V_O,
	vector<vector<Point2i>>& V_B, Mat& img);
void estimateMotion(Mat& I_O, Mat& I_B, vector<vector<vector<Point2i>>>& V_O_list,
	vector<vector<vector<Point2i>>>& V_B_list, vector<Mat&> &imgs);
