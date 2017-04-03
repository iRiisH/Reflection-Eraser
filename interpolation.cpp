#include "image.h"
#include "interpolation.h"
#include <time.h>
#define P 2
#include <opencv2/imgproc/imgproc.hpp>

// k-nearest neighbours interpolation
// UPDATE: too slow + bad results working with the motion fields
// it is thus preferable to use Delaunay triangulation
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

// Delaunay triangulation-based interpolation

void draw_subdiv(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{
#if 1
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);

	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		line(img, pt[0], pt[1], delaunay_color, 1, LINE_AA, 0);
		line(img, pt[1], pt[2], delaunay_color, 1, LINE_AA, 0);
		line(img, pt[2], pt[0], delaunay_color, 1, LINE_AA, 0);
	}
#else
	vector<Vec4f> edgeList;
	subdiv.getEdgeList(edgeList);
	for (size_t i = 0; i < edgeList.size(); i++)
	{
		Vec4f e = edgeList[i];
		Point pt0 = Point(cvRound(e[0]), cvRound(e[1]));
		Point pt1 = Point(cvRound(e[2]), cvRound(e[3]));
		line(img, pt0, pt1, delaunay_color, 1, LINE_AA, 0);
	}
#endif
}

void draw_subdiv_point(Mat& img, Point2f fp, Scalar color)
{
	circle(img, fp, 3, color, FILLED, LINE_8, 0);
}
void locate_point(Mat& img, Subdiv2D& subdiv, Point2f fp, Scalar active_color)
{
	int e0 = 0, vertex = 0;

	subdiv.locate(fp, e0, vertex);

	if (e0 > 0)
	{
		int e = e0;
		do
		{
			Point2f org, dst;
			if (subdiv.edgeOrg(e, &org) > 0 && subdiv.edgeDst(e, &dst) > 0)
			{
				line(img, org, dst, active_color, 3, LINE_AA, 0);
			}

			e = subdiv.getEdge(e, Subdiv2D::NEXT_AROUND_LEFT);
		} while (e != e0);
	}

	draw_subdiv_point(img, fp, active_color);
}
Subdiv2D createDelaunayTriangulation(vector<vector<Point2i>> v)
{
	assert(v.size() > 0);
	int m = v.size(), n = v[0].size();
	Rect rect(0, 0, n, m);

	Subdiv2D subdiv(rect);
	//vector<Point2i> sparseData = {};
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (v[i][j].x == 0 && v[i][j].y == 0)
				continue;
			Point2f p(j,i);
			subdiv.insert(p);
		}
	}
	return subdiv;
}



float dist(Point2f a, Point2f b)
{
	return sqrt(pow(a.x - b.x, 2.) + pow(a.y - b.y, 2.));
}

float value(vector<Point2f> l, vector<float> vals, Point2f p)
{
	float num = 0., denom = 0.;
	for (int i = 0; i < l.size(); i++)
	{
		float frac = 1. / pow(pow(l[i].x - p.x, 2.) + pow(l[i].y - p.y, 2.), P / 2.);
		denom += frac;
		num += vals[i] * frac;
	}
	return num / denom;
}

float interpolatedValue(Subdiv2D& subdiv, Point2f p, const Mat& img)
{
	vector<Point2f> vert;
	int e0, vertex;
	int cas = subdiv.locate(p, e0, vertex);
	int e = e0, i = 0, m = img.rows, n = img.cols;
	vector<Point2f> l;
	vector<float> vals;
	Point2f org, dst;
	switch (cas)
	{
	case Subdiv2D::PTLOC_INSIDE:
		do
		{
			Point2f org;
			subdiv.edgeOrg(e, &org);

			if (!(org.x < 0 || org.y < 0 || org.x > n || org.y > m))
			{
				l.push_back(org);
				vals.push_back(img.at<float>(org));
			}

			e = subdiv.getEdge(e, Subdiv2D::NEXT_AROUND_LEFT);
			i++;
		} while (e != e0);
		return value(l, vals, p);
		break;
	case Subdiv2D::PTLOC_ON_EDGE:
		subdiv.edgeOrg(e0, &org);
		subdiv.edgeDst(e0, &dst);
		if (!(org.x < 0 || org.y < 0 || org.x > n || org.y > m))
		{
			l.push_back(org);
			vals.push_back(img.at<float>(org));
		}
		if (!(dst.x < 0 || dst.y < 0 || dst.x > n || dst.y > m))
		{
			l.push_back(dst);
			vals.push_back(img.at<float>(dst));
		}
		return value(l, vals, p);
		break;
	case Subdiv2D::PTLOC_VERTEX:
		return img.at<float>(p);
		break;
	case Subdiv2D::PTLOC_OUTSIDE_RECT:
	case Subdiv2D::PTLOC_ERROR :
	default :
		cout << "An error happened during the interpolation" << endl;
		return 0.;
		break;
	}
}


void interpolateMotionField2(vector<vector<Point2i>> &v)
{
	assert(v.size() > 0);
	createDelaunayTriangulation(v);
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
	Subdiv2D subdiv = createDelaunayTriangulation(v);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			vx.at<float>(i,j) = interpolatedValue(subdiv, Point2f(j, i), vx);
			vy.at<float>(i, j) = interpolatedValue(subdiv, Point2f(j, i), vy);
		}
	}

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			v[i][j].y = (int)(round(vx.at<float>(i, j)));
			v[i][j].x = (int)(round(vy.at<float>(i, j)));
		}
	}
}
