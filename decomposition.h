#pragma once

#include "image.h"

float func_w1d(const Mat& I_t, const Mat& I_Oh, const Mat& I_Bh, const Mat& W_Ot, const Mat& W_Bt);
float func_w2d(const Mat& I_Bh);
float func_w3d(const Mat& I_Oh);
float L(const Mat& I_O, const Mat& I_Oh, const Mat& I_B, const Mat& I_Bh);
float L(const Mat& I_O, const Mat& I_B);