#pragma once

#include "image.h"

struct IntVec
{
	int x;
	int y;
};

float func_w1(const Mat& I_t, const Mat& I_O, const Mat& I_B, IntVec[][] V_Oth, IntVec[][] V_Bth, IntVec a);
float func_w2(const Mat& V_Oth);
float func_w3(const Mat& V_Bth);