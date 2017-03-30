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

#define P 4
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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


void detectEdges(string filename)
{
	src = imread("../"+filename);
	dst.create(src.size(), src.type());
	cvtColor(src, src_gray, CV_BGR2GRAY);
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
	CannyThreshold(0, 0);
	waitKey(0);
}

struct Fields {
	vector<vector<Point2i>> v1, v2;
};

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

	vector<Point2i> obj, scene;
	int m = F1.rows, n = F1.cols;
	for (int i = 0; i < m; i += 4)
	{
		for (int j = 0; j < n; j += 4)
		{
			if (F1.at<float>(i, j) != 0.)
			{
				const Point2f& fxy = flow.at<Point2f>(i, j);
				Point2i p2(cvRound(j + fxy.x), cvRound(i + fxy.y));
				//line(I1, Point2i (j, i), p2, Scalar(0, 255, 0));
				//circle(I1, Point2i(j, i), 1, Scalar(255, 0, 0));

				scene.push_back(Point2i (j, i));
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
			//arrowedLine(I1, scene[i], obj[i], Scalar(0, 255, 0));
			v1[scene[i].y][scene[i].x] = obj[i];
		}
		else
		{
			new_scene.push_back(scene[i]);
			new_obj.push_back(obj[i]);
		}
	}

	Scalar colors[3] = { Scalar(0, 0, 255), Scalar(51, 0, 102), Scalar(255, 128, 0) };
	int N = 1;

	for (int nb = 0; nb < N; nb++)
	{
		Mat new_mask;
		findHomography(new_scene, new_obj, new_mask, RANSAC);
		vector<Point2i> rms, rmo;
		for (int i = 0; i < new_mask.rows; i++)
		{
			if (new_mask.at<uchar>(i, 0) != 0)
			{
				//arrowedLine(I1, new_scene[i], new_obj[i], colors[nb]);
				v2[new_scene[i].y][new_scene[i].x] = new_obj[i];
			}
			else
			{
				rms.push_back(new_scene[i]);
				rmo.push_back(new_obj[i]);
			}

		}
		new_scene = rms; new_obj = rmo;
	}
	Fields f;
	f.v1 = v1;
	f.v2 = v2;
	return f;
//	imshow("I1", I1);
//	waitKey(0);

}

void nearestNeighbourWeightedInterpolation(Mat& img)
{
	assert(img.type() == CV_32F);
	int m = img.rows, n = img.cols;
	vector<Point2i> sparseData;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (img.at<float>(i, j) != 0.)
				sparseData.push_back(Point2i (i,j));
		}
	}
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (find(sparseData.begin(), sparseData.end(), Point2i (i,j)) != sparseData.end())
				continue;
			else
			{
				float sum = 0., norm = 0.;
				for (int k = 0; k < sparseData.size(); k++)
				{
					Point2i pK = sparseData[k];
					sum += img.at<float>(pK.x, pK.y) /
						pow(pow(float (pK.x - i), 2.) + pow(float(pK.y - j), 2.), (float)P / 2.);
					norm+= 1. /
						pow(pow(float (pK.x - i), 2.) + pow(float (pK.y - j), 2.), (float)P / 2.);
				}
				img.at<float>(i, j) = sum / norm;
			}	
		}
	}
}

void testInterpolation()
{
	Mat img = Mat::zeros(300, 300, CV_32F);
	Mat orig;
	img.copyTo(orig);
	srand(unsigned int (time(NULL)));
	int m = img.rows, n = img.cols;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			Point2i p(i, j);
			float r1 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			if (r1 < .005)
			{
				img.at<float>(p) = 255.*(float)cos((float)i / 25.)*cos((float)j / 25.);
			}
			orig.at<float>(p) = 255.*(float)cos((float)i / 25.)*cos((float)j / 25.);
		}
	}
	nearestNeighbourWeightedInterpolation(img);
	//imwrite("../results/original.bmp",orig);
	imwrite("../results/interpolation.bmp", img);
	//imshow("original", orig);
	//imshow("interpolation", img);
	
	waitKey(0);
}

void interpolateMotionField(vector<vector<Point2i>> &v)
{
	assert(v.size() > 0);
	int m = v.size(), n = v[0].size();
	Mat vx = Mat::zeros(m, n, CV_32F), vy = Mat::zeros(m, n, CV_32F);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			vx.at<float>(i, j) = (float)(v[i][j].x);
			vy.at<float>(i, j) = (float)(v[i][j].y);
		}
	}
	nearestNeighbourWeightedInterpolation(vx);
	nearestNeighbourWeightedInterpolation(vy);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			v[i][j].y = (int)(round(vx.at<float>(i, j)));
			v[i][j].x = (int)(round(vy.at<float>(i, j)));
		}
	}
}

void saveMotionField(const vector<vector<Point2i>> v, String filename)
{
	assert(v.size() > 0);
	ofstream file;
	file.open("../results/"+filename);
	int m = v.size (), n = v[0].size ();
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
	ifstream file("../results/"+filename);
	int i = 0, j = 0;
	if (file.is_open())
	{
		cout << "loading motion field at ../results/" + filename+" ...";
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
				case '1' :
				case '2' :
				case '3' :
				case '4' :
				case '5' :
				case '6' :
				case '7' :
				case '8' :
				case '9' :
				case '0' :
					num = c - '0';
					cnt = 10 * cnt + num;
					break;
				case ',' :
					v[i].push_back(p);
					v[i][j].x = cnt;
					cnt = 0;
					break;
				case ')' :
					v[i][j].y = cnt;
					cnt = 0;
					j++;
					break;
				default :
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
			v[0] =int( angle.at<float>(i,j) * 180. / M_PI / 2.);
			v[1] = 255;
			v[2] = int(255.*magnitude.at<float>(i, j)/max_mag);
			///cout << v[2] << endl;
		}
	}
	cvtColor(hsv, img, COLOR_HSV2BGR);

	imshow("motion field visualization", img);
	waitKey(0);
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

int main(int argc, char** argv)
{
	Mat I1 = imread("../edges1_red.png");
	Mat I2 = imread("../edges2_red.png");
	int m = I1.rows, n = I1.cols;
	Mat F1, F2, G1, G2;

	// uses canny edges detector
	//detectEdges("im2_red.png");

	// detects the sparse motion field of the edges, then interpolates it to the whole space
	Fields f = detectSparseMotion(I1, I2);
	
	interpolateMotionField(f.v1);
	interpolateMotionField(f.v2);
	saveMotionField(f.v1, "v1_red.txt");
	saveMotionField(f.v2, "v2_red.txt");
	
	/*// loads the previously saved motion field
	vector<vector<Point2i>> v1, v2;
	v1 = loadMotionField("v1_red.txt");
	v2 = loadMotionField("v2_red.txt");

	Mat img;
	I1.copyTo(img);
	///displayMotionField(v1, img);
	srand(time(NULL));
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			Point2i p1(j, i), p2(v1[i][j]);
			int test = rand() % 100;
			if (test <= 1)
				arrowedLine(I1, p1, p2, Scalar(255, 0, 0));

		}
	}

	
	imshow("I1", I1);
	waitKey(0);
	*/
	return 0;
}
