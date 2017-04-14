#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>
#include <string>
#include <fstream>
#include <time.h>

#include "image.h"
#include "interpolation.h"
#include "initialization.h"


using namespace cv;
using namespace std;

void detectEdges(const Mat& inArray, Mat& outArray)
{
	int kernel_size = 3;
	Mat temp, temp2;
	temp = temp2 = Mat::zeros(inArray.size(), inArray.type());
	outArray = Mat::zeros(inArray.size(), inArray.type());
	blur(inArray, temp, Size(3, 3));
	Canny(temp, temp2, EDGES_THRESHOLD, EDGES_THRESHOLD*EDGES_RATIO, kernel_size);
	inArray.copyTo(outArray, temp2);
}

void detectEdges(const vector<Mat>& inArray, vector<Mat>& outArray)
{
	assert(inArray.size() == outArray.size());
	int N = inArray.size();
	for (int i = 0; i < N; i++)
	{
		detectEdges(inArray[i], outArray[i]);
	}
}

Fields detectSparseMotion(Mat& I1, Mat& I2)
{
	Mat G1, G2;
	cvtColor(I1, G1, COLOR_BGR2GRAY);
	cvtColor(I2, G2, COLOR_BGR2GRAY);
	Mat F1, F2;
	G1.convertTo(F1, CV_32F);
	G2.convertTo(F2, CV_32F);

	Mat flow;
	calcOpticalFlowFarneback(F1, F2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

	int m = I1.rows, n = I1.cols;

	vector<Point2i> obj, scene;
	for (int i = 0; i < m; i += 4)
	{
		for (int j = 0; j < n; j += 4)
		{
			if (F1.at<float>(i, j) != 0.)
			{
				const Point2f& fxy = flow.at<Point2f>(i, j);
				Point2i p2(cvRound(j + fxy.x), cvRound(i + fxy.y));
				//arrowedLine(img, Point2i (j, i), p2, Scalar(255, 0, 0));
				//circle(I1, Point2i(j, i), 1, Scalar(255, 0, 0));

				scene.push_back(Point2i(j, i));
				obj.push_back(p2);
			}
		}
	}
	
	Mat mask;
	findHomography(scene, obj, mask, RANSAC);

	vector <vector<Point2i>> v1, v2;
	// initialize motion fields with zeros
	for (int i = 0; i < m; i++)
	{
		vector<Point2i> temp1, temp2;
		for (int j = 0; j < n; j++)
		{
			temp1.push_back(Point2i(0, 0));
			temp2.push_back(Point2i(0, 0));
		}
		v1.push_back(temp1);
		v2.push_back(temp2);
	}
	vector<Point2i> new_obj, new_scene;
	for (int i = 0; i < mask.rows; i++)
	{
		if (mask.at<uchar>(i, 0) != 0)
		{
			//arrowedLine(img, scene[i], obj[i], Scalar(0, 255, 0));
			v1[scene[i].y][scene[i].x] = Point2i(obj[i].x-scene[i].x, obj[i].y -scene[i].y);
		}
		else
		{
			new_scene.push_back(scene[i]);
			new_obj.push_back(obj[i]);
		}
	}

	Scalar colors[3] = { Scalar(0, 0, 255), Scalar(51, 0, 102), Scalar(255, 128, 0) };

	Mat new_mask;
	if (!new_scene.empty())
	{
		findHomography(new_scene, new_obj, new_mask, RANSAC);
		for (int i = 0; i < new_mask.rows; i++)
		{
			if (new_mask.at<uchar>(i, 0) != 0)
			{
				//arrowedLine(img, new_scene[i], new_obj[i], Scalar(0, 0, 255));
				v2[new_scene[i].y][new_scene[i].x] = Point2i(new_obj[i].x - new_scene[i].x, new_obj[i].y - new_scene[i].y);
			}
		}
	}
	else
	{
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
				v2[i][j] = Point2i(v1[i][j].x, v1[i][j].y);
		}
	}
	Fields f;
	f.v1 = v1;
	f.v2 = v2;

	return f;
	//	imshow("I1", I1);
	//	waitKey(0);

}

void saveMotionField(const vector<vector<Point2i>> v, String filename)
{
	assert(v.size() > 0);
	ofstream file;
	file.open("../results/" + filename);
	int m = v.size(), n = v[0].size();
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
			file << "(" << v[i][j].x << ", " << v[i][j].y << ") ";
		file << "\n";
	}
	file.close();
}

vector<vector<Point2i>> loadMotionField(String filename)
{
	vector<vector<Point2i>> v;
	string line;
	ifstream file("../results/" + filename);
	int i = 0, j = 0;
	if (file.is_open())
	{
		cout << "loading motion field at ../results/" + filename + " ...";
		while (getline(file, line))
		{
			vector<Point2i> new_line;
			v.push_back(new_line);
			j = 0;
			int x, y;
			int cnt = 0;
			for (int k = 0; k < line.size(); k++)
			{
				int num;
				Point2i p;
				char c = line.at(k);
				switch (c)
				{
				case '1':
				case '2':
				case '3':
				case '4':
				case '5':
				case '6':
				case '7':
				case '8':
				case '9':
				case '0':
					num = c - '0';
					cnt = 10 * cnt + num;
					break;
				case ',':
					v[i].push_back(p);
					v[i][j].x = cnt;
					cnt = 0;
					break;
				case ')':
					v[i][j].y = cnt;
					cnt = 0;
					j++;
					break;
				default:
					break;
				}
			}
			i++;
		}
		file.close();
		cout << "done" << endl;
	}

	else cout << "Unable to open file" << endl;;
	return v;
}

void displayMotionField(const vector<vector<Point2i>> v, Mat& img)
{
	assert(v.size() > 0);
	int m = v.size(), n = v[0].size();
	Mat x_coord = Mat::zeros(m, n, CV_32F), y_coord = Mat::zeros(m, n, CV_32F);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			x_coord.at<float>(i, j) = float(v[i][j].x);
			y_coord.at<float>(i, j) = float(v[i][j].y);
		}
	}
	Mat magnitude, angle;
	cartToPolar(x_coord, y_coord, magnitude, angle);
	img = Mat::zeros(m, n, CV_8UC3);
	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV);

	float max_mag = 0.;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			float new_mag = magnitude.at<float>(i, j);
			max_mag = new_mag > max_mag ? new_mag : max_mag;
		}
	}
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			Vec3b &v = hsv.at<Vec3b>(i, j);
			v[0] = int(angle.at<float>(i, j) * 180. / M_PI / 2.);
			v[1] = 255;
			v[2] = int(255.*magnitude.at<float>(i, j) / max_mag);
			///cout << v[2] << endl;
		}
	}
	cvtColor(hsv, img, COLOR_HSV2BGR);

	//imshow("motion field visualization", img);
	//waitKey(0);
}

void estimateInitialBackground(vector<vector<Point2i>> v_b, const Mat& I1, const Mat& I2)
{
	int m = I1.rows, n = I1.cols;
	Mat img = Mat::zeros(m, n, CV_32F);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			Point2i v_ij = v_b[i][j];
			if (i + v_ij.y >= 0 && i + v_ij.y < m && j + v_ij.x >= 0 && j + v_ij.x < n)
			{
				img.at<float>(i, j) = I2.at<float>(i + v_ij.y, j + v_ij.x);
				img.at<float>(i, j) /= 2.;
			}
			else
				img.at<float>(i, j) = 0.;
		}
	}
	normalize(img, img, 0., 1., NORM_MINMAX);
	imshow("img", img);
	waitKey(0);
}

void loadImages(vector<Mat>& images, Mat& im_ref)
{
	String path = PATH;
	assert(images.size() == N_IMGS);
	cout << "Loading data..." << endl;
	int i = 0;
	for (; i < (N_IMGS + 1) / 2; i++)
		images[i] = imread(path + "im" + to_string(i + 1) + ".png");
	im_ref = imread(path + "im" + to_string(i+1) + ".png");

	for (; i < N_IMGS; i++)
		images[i] = imread(path + "im" + to_string(i + 2) + ".png");

	if (RESIZE)
	{
		for (int i = 0; i < N_IMGS; i++)
			resize(images[i], images[i], Size2i(0, 0), RESIZE_RATIO, RESIZE_RATIO);
		resize(im_ref, im_ref, Size2i(0, 0), RESIZE_RATIO, RESIZE_RATIO);
	}
}

void initialize(vector<Mat>& images, Mat& img_ref, vector<Fields>& motionFields, Mat& I_O, Mat& I_B,
	vector<vector<vector<Point2i>>>& V_O_list, vector<vector<vector<Point2i>>>& V_B_list)
{
	vector<Mat> edges(N_IMGS);
	Mat edges_ref;
	int m = img_ref.rows, n = img_ref.cols;
	V_O_list = vector<vector<vector<Point2i>>>(N_IMGS),
		V_B_list = vector<vector<vector<Point2i>>>(N_IMGS);
	cout << "Computing edges..." << endl;
	detectEdges(images, edges);
	detectEdges(img_ref, edges_ref);
	//imshow("test", edges_ref);
	//waitKey(0);
	vector<Fields> f(N_IMGS);
	Mat warpedImages[N_IMGS];// warpedReflections[N_IMGS];
	cout << "Detecting edge flow..." << endl;
	for (int i = 0; i < N_IMGS; i++)
		f[i] = detectSparseMotion(edges[i], edges_ref);


	cout << "Interpolating motion field..." << endl;
	for (int i = 0; i < N_IMGS; i++)
	{
		interpolateMotionField2(f[i].v1);
		interpolateMotionField2(f[i].v2);
		V_O_list[i] = f[i].v2;
		V_B_list[i] = f[i].v1;
		warpImage<Vec3b>(images[i], warpedImages[i], f[i].v1);
//		warpImage<Vec3b>(images[i], warpedReflections[i], f[i].v2);
	}

	cout << "Computing initial values for Io and Ib..." << endl;
	Mat res = Mat::zeros(img_ref.size(), img_ref.type());
	Mat reflect = Mat::zeros(img_ref.size(), img_ref.type());
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			vector<int> l;
			for (int k = 0; k < N_IMGS; k++)
			{
				Vec3b val = warpedImages[k].at<Vec3b>(i, j);
				l.push_back(val[0] + val[1] + val[2]);
			}
			int ind = min_ind<int>(l);
			Vec3b val_temp = warpedImages[ind].at<Vec3b>(i, j);
			Vec3b val = img_ref.at<Vec3b>(i, j);
			Vec3b min_val = (val[0] + val[1] + val[2] <
				val_temp[0] + val_temp[1] + val_temp[2]) ? val : val_temp;
			res.at<Vec3b>(i, j) = min_val;
			Vec3b ref_val;
			Vec3b orig = img_ref.at<Vec3b>(i, j);
			ref_val[0] = abs(min_val[0] - orig[0]);
			ref_val[1] = abs(min_val[1] - orig[1]);
			ref_val[2] = abs(min_val[2] - orig[2]);
			if (min_val == Vec3b(0, 0, 0))
				ref_val = Vec3b(0, 0, 0);
			reflect.at<Vec3b>(i, j) = ref_val;
		}
	}
	motionFields = f;
	I_O = res;
	I_B = reflect;
	//imshow("Initialisation du reflet", reflect);
	//imshow("Initialisation du fond", res);
	//imwrite("../result.png", reflect);
	//waitKey(0);
}

void zero_initialize(int m, int n,vector<Fields>& motionFields, Mat& I_O, Mat& I_B,
	vector<vector<vector<Point2i>>>& V_O_list, vector<vector<vector<Point2i>>>& V_B_list)
	// for test purposes
{
	V_O_list = vector<vector<vector<Point2i>>>(N_IMGS);
	V_B_list = vector<vector<vector<Point2i>>>(N_IMGS);
	for (int k = 0; k < N_IMGS; k++)
	{
		V_O_list[k] = vector<vector<Point2i>>(m);
		V_B_list[k] = vector<vector<Point2i>>(m);
		for (int i = 0; i < m; i++)
		{
			V_O_list[k][i] = vector<Point2i>(n);
			V_B_list[k][i] = vector<Point2i>(n);
			for (int j = 0; j < n; j++)
			{
				V_O_list[k][i][j] = Point2i(0, 0);
				V_B_list[k][i][j] = Point2i(0, 0);
			}
		}
	}
	I_O = Mat::zeros(m, n, CV_8UC3);
	I_B = Mat::zeros(m, n, CV_8UC3);
}