#include "image.h"
#include "interpolation.h"

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
				sparseData.push_back(Point2i(i, j));
		}
	}
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (find(sparseData.begin(), sparseData.end(), Point2i(i, j)) != sparseData.end())
				continue;
			else
			{
				float sum = 0., norm = 0.;
				for (int k = 0; k < sparseData.size(); k++)
				{
					Point2i pK = sparseData[k];
					sum += img.at<float>(pK.x, pK.y) /
						pow(pow(float(pK.x - i), 2.) + pow(float(pK.y - j), 2.), (float)P / 2.);
					norm += 1. /
						pow(pow(float(pK.x - i), 2.) + pow(float(pK.y - j), 2.), (float)P / 2.);
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
	srand(unsigned int(time(NULL)));
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