#include "image.h"
#include "initialization.h"
#include "decomposition.h"
#include "motion.h"
#include "interpolation.h"

#include <time.h>



int main(int argc, char** argv)
{
	vector<Mat> imgs(N_IMGS);
	Mat img_ref;
	loadImages(imgs, img_ref);

	vector<Fields> f(N_IMGS);
	Mat I_O, I_B;
	initialize(imgs, img_ref, f, I_O, I_B);

	int N = 4;
	for (int k = 0; k < N; k++)
	{
		//decompose(I_O, I_B);
	}

	return 0;
}
