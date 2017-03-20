#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>

using namespace cv;
using namespace std;


/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;


int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

void CannyThreshold(int, void*)
{
	/// Reduce noise with a kernel 3x3
	blur(src_gray, detected_edges, Size(3, 3));
	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);
	src.copyTo(dst, detected_edges);
	imshow(window_name, dst);
}


void detectEdges()
{
	src = imread("../im1.png");
	dst.create(src.size(), src.type());
	cvtColor(src, src_gray, CV_BGR2GRAY);
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
	CannyThreshold(0, 0);
	waitKey(0);
}

void detectSparseMotion(Mat& I1, Mat& I2)
{
	Mat G1, G2;

	cvtColor(I1, G1, COLOR_BGR2GRAY);
	cvtColor(I2, G2, COLOR_BGR2GRAY);
	Mat F1, F2;
	G1.convertTo(F1, CV_32F);
	G2.convertTo(F2, CV_32F);

	Mat flow;
	calcOpticalFlowFarneback(F1, F2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

	vector<Point2f> obj, scene;
	int m = F1.rows, n = F1.cols;
	for (int i = 0; i < n; i += 4)
	{
		for (int j = 0; j < m; j += 4)
		{
			Point2f p1(i, j);
			if (F1.at<float>(p1) != 0.)
			{
				const Point2f& fxy = flow.at<Point2f>(p1);
				Point2f p2(cvRound(i + fxy.x), cvRound(j + fxy.y));
				//line(I1, p1, p2, Scalar(0, 255, 0));
				circle(I1, p1, 1, Scalar(255, 0, 0));

				scene.push_back(p1);
				obj.push_back(p2);
			}
		}
	}
	Mat mask;
	findHomography(scene, obj, mask, RANSAC);

	vector<Point2f> new_obj, new_scene;
	for (int i = 0; i < mask.rows; i++)
	{
		if (mask.at<uchar>(i, 0) != 0)
			arrowedLine(I1, scene[i], obj[i], Scalar(0, 255, 0));
		else
		{
			new_scene.push_back(scene[i]);
			new_obj.push_back(obj[i]);
		}
	}


	Scalar colors[3] = { Scalar(0, 0, 255), Scalar(51, 0, 102), Scalar(255, 128, 0) };
	int N = 3;

	for (int nb = 0; nb < N; nb++)
	{
		Mat new_mask;
		cout << new_scene.size() << " - " << new_obj.size() << endl;
		findHomography(new_scene, new_obj, new_mask, RANSAC);
		vector<Point2f> rms, rmo;
		for (int i = 0; i < new_mask.rows; i++)
		{
			if (new_mask.at<uchar>(i, 0) != 0)
				arrowedLine(I1, new_scene[i], new_obj[i], colors[nb]);
			else
			{
				rms.push_back(new_scene[i]);
				rmo.push_back(new_obj[i]);
			}

		}
		new_scene = rms; new_obj = rmo;
	}
	imshow("I1", I1);
	waitKey(0);
}

int main(int argc, char** argv)
{
	Mat I1 = imread("../edges1.png");
	Mat I2 = imread("../edges2.png");
	
	//edges();
	detectSparseMotion(I1, I2);

	return 0;
}
