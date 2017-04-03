#include "image.h"
#include "initialization.h"
#include "decomposition.h"
#include "motion.h"
#include "interpolation.h"

#include <time.h>

int main(int argc, char** argv)
{
	Mat I1 = imread("../edges1_red.png");
	Mat I2 = imread("../edges2_red.png");
	Mat I1_color = imread("../im1_red.png");
	Mat I2_color = imread("../im2_red.png");
	int m = I1.rows, n = I1.cols;
	Mat F1, F2, G1, G2;
	I1.convertTo(F1, CV_32F);
	// uses canny edges detector
	//detectEdges("im2_red.png");

	// detects the sparse motion field of the edges, then interpolates it to the whole space
	Fields f = detectSparseMotion(I1, I2);
	/*for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (f.v1[i][j].x != 0 && f.v1[i][j].y != 0)
			{
				arrowedLine(I1, Point2f(j, i), Point2f(j+f.v1[i][j].x, i+f.v1[i][j].y), Scalar(0, 0, 255));
			}
		}
	}*/
	//imshow("I1", I1);
	//waitKey(0);
	Subdiv2D subdiv = createDelaunayTriangulation(f.v1);
	//draw_subdiv(I1, subdiv, Scalar(255, 255, 255));

	interpolateMotionField2(f.v1);

	Mat img;
	I1.copyTo(img);

	warpImage<Vec3b>(I1_color, img, f.v1);
	imshow("img", img);
	waitKey(0);
	//displayMotionField(f.v1, I1);
	//imwrite("v1.png", I1);

	return 0;
}
