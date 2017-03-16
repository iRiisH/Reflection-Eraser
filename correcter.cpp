#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <queue>
#include "image.h"

using namespace std;
using namespace cv;

vector<Point> harris(const Image<float> &I, int n)
{
	vector<Point> res;
	int blockSize = 10;
	int apertureSize = 3;
	double k = 0.04;

	Mat out, out_norm, out_norm_scaled;
	out_norm = Mat::zeros(I.size(), CV_32FC1);
	cornerHarris(I, out_norm, blockSize, apertureSize, k, BORDER_DEFAULT);

	float threshold = 2000;
	for (int i = n; i < out_norm.rows - n; i++)
	{
		for (int j = n; j < out_norm.cols - n; j++)
		{
			if (out_norm.at<float>(i, j) > threshold
				&& out_norm.at<float>(i, j) > out_norm.at<float>(i - 1, j)
				&& out_norm.at<float>(i, j) > out_norm.at<float>(i, j - 1)
				&& out_norm.at<float>(i, j) > out_norm.at<float>(i + 1, j)
				&& out_norm.at<float>(i, j) > out_norm.at<float>(i, j + 1)
				&& out_norm.at<float>(i, j) > out_norm.at<float>(i + 1, j + 1)
				&& out_norm.at<float>(i, j) > out_norm.at<float>(i + 1, j - 1)
				&& out_norm.at<float>(i, j) > out_norm.at<float>(i - 1, j - 1)
				&& out_norm.at<float>(i, j) > out_norm.at<float>(i - 1, j + 1))
			{
				res.push_back(Point(j, i)); 
			}
		}
	}
	return res;
}

struct Paire
{
	Point m1;
	Point m2;
};

vector<Paire> grains(const Image<float> &I1, const Image<float> &I2, int n, float s1, float s2)
{
	vector<Point> har = harris(I1, n);
	vector<Paire> res;
	cout << har.size() << endl;
	for (int i = 0; i < har.size(); i++)
	{
		double ncc_max = -1., ncc_max_2 = -1.;
		if (har[i].y < I2.rows)
		{
			Point m2 = Point(0, har[i].y), m2_prime = Point(0, har[i].y);
			for (int u = n; u < I2.cols - n; u++)
			{
				double ncc = NCC(I1, har[i], I2, Point(u, har[i].y), n);
				if (ncc > ncc_max)
				{
					ncc_max_2 = ncc_max;
					ncc_max = ncc;
					m2_prime = m2;
					m2 = Point(u, har[i].y);
				}
			}
			if (ncc_max > s1 && (1. + NCC(I1, har[i], I2, m2, n)) / (1. + NCC(I1, har[i], I2, m2_prime, n)) > s2)
			{
				Paire new_entry; new_entry.m1 = har[i]; new_entry.m2 = m2;
				res.push_back(new_entry);
			}
		}
	}
	return res;
	
}

Image<int> disparites (const Image<float> &I1, const Image<float> &I2, int n, float s1, float s2, float s3, bool corrected)
{
	Image<int> d(I1.width(), I1.height() , CV_32S);
	int d_min = I1.rows * I1.cols;
	for (int i = 0; i < d.width (); i++)
	{
		for (int j = 0; j < d.height (); j++)
		{
			d(i, j) = I1.rows * I1.cols;
		}
	}
	vector<Paire> tab_grains = grains(I1, I2, n, s1, s2);
	queue<Point> file;
	for (int i = 0; i < tab_grains.size(); i++)
	{
		file.push(tab_grains[i].m1);
		d(tab_grains[i].m1) = tab_grains[i].m2.x - tab_grains[i].m1.x;
	}
	while (!file.empty())
	{
		Point m1 = file.front();
		file.pop();
		vector<Point> voisins;
		if (m1.x > n+1)
		{
			if (m1.y > n+1)
				voisins.push_back(Point(m1.x - 1, m1.y-1));
			voisins.push_back(Point(m1.x - 1, m1.y));
			if (m1.y < I1.rows-n-1)
				voisins.push_back(Point(m1.x - 1, m1.y+1));
		}
		if (m1.x < I1.cols - n - 1)
		{
			if (m1.y > n + 1)
				voisins.push_back(Point(m1.x + 1, m1.y - 1));
			voisins.push_back(Point(m1.x + 1, m1.y));
			if (m1.y < I1.rows - n - 1)
				voisins.push_back(Point(m1.x + 1, m1.y + 1));
		}
		if (m1.y > n + 1) {
			voisins.push_back(Point(m1.x, m1.y - 1));
		}
		if (m1.y < I1.rows - n - 1) {
			voisins.push_back(Point(m1.x, m1.y + 1));
		}

		for (int i = 0; i < voisins.size(); i++)
		{
			if (d(voisins[i]) == I1.rows * I1.cols)
			{
				Point q = voisins[i];
				double max_ncc = -1.1;
				int k_max = -1;
				for (int k = -1; k <= 1; k++)
				{
					int d_q = d(m1) + k;
					double ncc = NCC(I1, q, I2, Point(q.x + d_q, q.y), n);
					if (ncc > max_ncc)
					{
						max_ncc = ncc;
						k_max = k;
					}
				}
				if (max_ncc > s3)
				{
					d(q) = d(m1) + k_max;
					if (d(q) < d_min)
						d_min = d(q);
					file.push(q);
				}
			}
		}
	}
	if (corrected)
	{
		for (int i = 0; i < d.width(); i++)
		{
			for (int j = 0; j < d.height(); j++)
			{
				if (d(i, j) == I1.rows * I1.cols)
					d(i, j) = d_min;
			}
		}
	}

	return d;
}

// Sauve un maillage triangulaire dans un fichier ply.
// v: sommets (x,y,z), f: faces (indices des sommets), col: couleurs (par sommet) 
bool savePly(const string& name, const vector<Point3f>& v, const vector<Vec3i>& f, const vector<Vec3b>& col) {
	assert(v.size() == col.size());
	ofstream out(name.c_str());
	if (!out.is_open()) {
		cout << "Cannot save " << name << endl;
		return false;
	}
	out << "ply" << endl
		<< "format ascii 1.0" << endl
		<< "element vertex " << v.size() << endl
		<< "property float x" << endl
		<< "property float y" << endl
		<< "property float z" << endl
		<< "property uchar red" << endl
		<< "property uchar green" << endl
		<< "property uchar blue" << endl
		<< "element face " << f.size() << endl
		<< "property list uchar int vertex_index" << endl
		<< "end_header" << endl;
	for (int i = 0; i<v.size(); i++)
		out << v[i].x << " " << v[i].y << " " << v[i].z << " " << int(col[i][2]) << " " << int(col[i][1]) << " " << int(col[i][0]) << " " << endl;
	for (int i = 0; i<f.size(); i++)
		out << "3 " << f[i][0] << " " << f[i][1] << " " << f[i][2] << endl;
	out.close();
	return true;
}

int main(int argc, char **argv)
{
	Mat I1 = imread("../face00R.tif", 1), I2 = imread("../face01R.tif", 1);
	Mat G1, G2; cvtColor(I1, G1, CV_BGR2GRAY); cvtColor(I2, G2, CV_BGR2GRAY);	
	Mat F1, F2;
	G1.convertTo(F1, CV_32F);
	G2.convertTo(F2, CV_32F);

	/*vector<Point> har = harris(F1, 10);
	cout << har.size() << endl;
	for (int i = 0; i < har.size(); i++)
		circle(I1, har[i], 3, Scalar(0, 0, 255), -1, 8);
	*/
	
	/*
	vector<Paire> tab_grains = grains(F1, F2, 10, 0.8, 1.1);
	for (int i = 0; i < tab_grains.size(); i++)
	{
		circle(I1, tab_grains[i].m1, 3, Scalar(0, 0, 255), -1, 8);
		circle(I2, tab_grains[i].m2, 3, Scalar(0, 0, 255), -1, 8);
	}*/
	Image<int> d = disparites(F1, F2, 3, 0.8, 1.1, 0.6, false);

	imshow("D", d.greyImage());
	//imshow("I1", I1);
	//imshow("I2", I2);
	
	Mat res (d.height(), d.width(), CV_32F);
	for (int i = 0; i < F1.rows; i++)
	{
		int last_d = F1.cols * F1.rows;
		for (int j = F1.cols-1; j >= 0; j--)
		{
			Point m1 = Point(j, i);
			if (d(m1) == F1.cols * F1.rows)
			{
				if (last_d == F1.cols * F1.rows)
					res.at<float>(m1) = F1.at<float>(m1);
				else // dans ce cas pas de correspondance
				{
					float p1 = F1.at<float>(m1), p2 = F2.at<float>(Point(j + last_d, i));
					float min = (p1 > p2) ? p2 : p1;

					res.at<float>(m1) = min;
				}
			}
			else
			{
				last_d = d(m1);
				float p1 = F1.at<float>(m1), p2 = F2.at<float>(Point(j + last_d, i));
				float min = (p1 > p2) ? p2 : p1;
				res.at<float>(m1) = min;//(F1.at<float>(m1) + F2.at<float>(Point(j + d(m1), i))) / 2.;
			}
		}
	}
	imshow("G1", G1);
	imshow("G2", G2);
	cv::normalize(res, res, 0, 1, cv::NORM_MINMAX);
	imshow("Result", res);

	vector<Point3f> vertices;
	vector<Vec3i> triangles;
	vector<Vec3b> couleurs;
	for (int i = 0; i < F1.rows ; i++)
	{
		for (int j = 0; j < F1.cols; j++)
		{
			Point m1 = Point(j, i);
			vertices.push_back(Point3f(j, i, 100000. / (d(m1)+100.)));
			couleurs.push_back(I1.at<Vec3b>(m1));
		}
	}
	for (int i = 0; i < F1.rows-1; i++)
	{
		for (int j = 0; j < F1.cols-1; j++)
		{
			Point m1 = Point(j, i);
			triangles.push_back( Vec3i(i*F1.cols + j, i*F1.cols+j+1, (i+1)*F1.cols + j) );
			triangles.push_back(Vec3i(i*F1.cols + j + 1, (i+1)*F1.cols + j + 1, (i + 1)*F1.cols + j));
		}
	}
	savePly("test.ply", vertices, triangles, couleurs);

	waitKey(0);
	return 0;
}