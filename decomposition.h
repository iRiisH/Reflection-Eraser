#pragma once

#include "image.h"

//float func_w1d(const Mat& I_t, const Mat& I_Oh, const Mat& I_Bh, const Mat& W_Ot, const Mat& W_Bt);
//float func_w2d(const Mat& I_Bh);
//float func_w3d(const Mat& I_Oh);
//float L(const Mat& I_O, const Mat& I_Oh, const Mat& I_B, const Mat& I_Bh);
float L(const Mat& I_O, const Mat& I_B);
float objective1(const Mat& I_O, const Mat& I_B, int k);
Mat& imgToVec(const Mat& img);
void vecToImg(const double* vec, const int m, const int n, bool is_I_O);
void solve_O(int channel);
void solve_B(int channel);
void decompose(Mat& I_O, Mat& I_B, vector<vector<vector<Point2i>>>& V_O_list,
	vector<vector<vector<Point2i>>>& V_B_list, vector<Mat>& imgs, Mat& img_ref);
