#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

struct Fields
{
	vector<vector<Point2i>> v1, v2;
};

void detectEdges(const Mat& inArray, Mat& outArray);
void detectEdges(const vector<Mat>& inArray, vector<Mat>& outArray);
Fields detectSparseMotion(Mat& I1, Mat& I2);
void saveMotionField(const vector<vector<Point2i>> v, String filename);
vector<vector<Point2i>> loadMotionField(String filename);
void displayMotionField(const vector<vector<Point2i>> v, Mat& img);
void estimateInitialBackground(vector<vector<Point2i>> v_b, const Mat& I1, const Mat& I2);
void loadImages(vector<Mat>& images, Mat& im_ref);
void initialize(vector<Mat>& images, Mat& img_ref, vector<Fields>& motionFields, Mat& I_O, Mat& I_B);