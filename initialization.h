#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

void CannyThreshold(int, void*);
void detectEdges(string filename);
struct Fields {
	vector<vector<Point2i>> v1, v2;
};
Fields detectSparseMotion(Mat& I1, Mat& I2);
void saveMotionField(const vector<vector<Point2i>> v, String filename);
vector<vector<Point2i>> loadMotionField(String filename);
void displayMotionField(const vector<vector<Point2i>> v, Mat& img);
void estimateInitialBackground(vector<vector<Point2i>> v_b, const Mat& I1, const Mat& I2);
