#include "image.h"
#include "initialization.h"
#include "decomposition.h"
#include "motion.h"
#include "interpolation.h"

#include <time.h>

#define EDGES_THRESHOLD 45
#define N_IMGS 4

int main(int argc, char** argv)
{
	vector<Mat> imgs(N_IMGS), edges(N_IMGS);
	Mat img_ref;
	loadImages(imgs, img_ref);
	detectEdges(imgs, edges, EDGES_THRESHOLD);

	Fields f[N_IMGS];
	Mat warpedImages[N_IMGS];
	for (int i = 0; i < N_IMGS; i++)
	{
		f[i] = detectSparseMotion(imgs[i], img_ref);
		interpolateMotionField2(f[i].v1);
		interpolateMotionField2(f[i].v2);
		warpImage<Vec3b>(imgs[i], warpedImages[i], f[i].v1);
	}

	int m = img_ref.rows, n = img_ref.cols;
	Mat res = Mat::zeros(img_ref.size(), img_ref.type());
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			vector<int> l;
			for (int k = 0; k < N_IMGS; k++)
			{
				Vec3b val = imgs[k].at<Vec3b>(i, j);
				l.push_back(val[0] + val[1] + val[2]);
			}
			int ind = min<int>(l);
			res.at<Vec3b>(i, j) = imgs[ind].at<Vec3b> (i, j);
		}
	}
	imshow("Initialisation", res);
	imwrite("../result.png", res);
	waitKey(0);

	return 0;
}
