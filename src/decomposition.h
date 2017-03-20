#pragma once

#include "image.h"

float func_w1(const Mat& I_t, const Mat& I_Oh, const Mat& I_Bh, const Mat& W_Ot,
	const Mat& W_Ot, const Mat& W_Bt);
float func_w2(const Mat& I_Bh);
float func_w3(const Mat& I_Oh);
float L(I_O, I_Oh, I_B, I_Bh);