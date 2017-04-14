#include "image.h"
#include "initialization.h"
#include "decomposition.h"
#include "motion.h"
#include "interpolation.h"

#include <time.h>

void testSparseMotion();
void testTriangulation();
void testInterpol();
void testInitialisation();
void testSimpleAlg();

int main(int argc, char** argv)
{
	//testSparseMotion();
	//testTriangulation();
	//testInterpol();
	//testInitialisation();
	testSimpleAlg();

	return 0;
}

void testSparseMotion()
{
	vector<Mat> imgs(N_IMGS);
	Mat img_ref;
	loadImages(imgs, img_ref);
	Mat edges, edges_ref;
	detectEdges(imgs[0], edges);
	detectEdges(img_ref, edges_ref);
	Fields f = detectSparseMotion(edges, edges_ref);
	
	int m = img_ref.rows, n = img_ref.cols;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (f.v1[i][j].x != 0 || f.v1[i][j].y != 0)
			{
				arrowedLine(edges, Point2i(j, i),
					Point2i(j + f.v1[i][j].x, i + f.v1[i][j].y), Scalar(255, 0, 0));
			}
			if (f.v2[i][j].x != 0 || f.v2[i][j].y != 0)
			{
				arrowedLine(edges, Point2i(j, i),
					Point2i(j + f.v2[i][j].x, i + f.v2[i][j].y), Scalar(0, 0, 255));
			}
		}
	}

	imshow("Test de triangulation", edges);
	//imwrite("../test_triangulation.png", edges);
	waitKey(0);
}

void testTriangulation()
{
	vector<Mat> imgs(N_IMGS);
	Mat img_ref;
	loadImages(imgs, img_ref);
	Mat edges, edges_ref;
	detectEdges(imgs[0], edges);
	detectEdges(img_ref, edges_ref);
	Fields f = detectSparseMotion(edges, edges_ref);
	Subdiv2D subdiv = createDelaunayTriangulation(f.v1);
	draw_subdiv(edges, subdiv, Scalar(255, 255, 255));
	Point2f p(300, 150);
	locate_point(edges, subdiv, p, Scalar(0, 0, 255));

	imshow("Test de triangulation", edges);
	//imwrite("../test_triangulation.png", edges);
	waitKey(0);
}

void testInterpol()
{
	vector<Mat> imgs(N_IMGS);
	Mat img_ref;
	loadImages(imgs, img_ref);
	Mat edges, edges_ref;
	detectEdges(imgs[0], edges);
	detectEdges(img_ref, edges_ref);
	Fields f = detectSparseMotion(edges, edges_ref);
	interpolateMotionField(f.v1);
	displayMotionField(f.v1, img_ref);
	imshow("Test d'interpolation", img_ref);
	imwrite("../test_interpolation.png", img_ref);
	waitKey();
}

void testInitialisation()
{
	vector<Mat> imgs(N_IMGS);
	Mat img_ref;
	loadImages(imgs, img_ref);
	vector<Fields> f(N_IMGS);
	Mat I_O, I_B;
	vector<vector<vector<Point2i>>> V_O_list, V_B_list;
	int m = img_ref.rows, n = img_ref.cols;
	zero_initialize(m, n, f, I_O, I_B, V_O_list, V_B_list);
	initialize(imgs, img_ref, f, I_O, I_B, V_O_list, V_B_list);
	imshow("Initialisation du reflet", I_O);
	imshow("Initialisation du fond", I_B);
	waitKey(0);
}

void testSimpleAlg ()
{
	vector<Mat> imgs(N_IMGS);
	Mat img_ref;
	loadImages(imgs, img_ref);
	vector<Fields> f(N_IMGS);
	Mat I_O, I_B;
	vector<vector<vector<Point2i>>> V_O_list, V_B_list;
	int m = img_ref.rows, n = img_ref.cols;
	initialize(imgs, img_ref, f, I_O, I_B, V_O_list, V_B_list);

	int N = 4;
	for (int k = 0; k < N; k++)
	{
		decompose(I_O, I_B, V_O_list, V_B_list, imgs, img_ref);
		estimateMotion(I_O, I_B, V_O_list, V_B_list, imgs);
	}
	imwrite("../four_iter.png", I_O);
	imwrite("../four_iter_ref.png", I_B);
}