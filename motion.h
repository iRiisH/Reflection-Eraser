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