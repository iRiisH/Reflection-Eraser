#include "image.h"


void cannyEdgeDetector(const Mat& img)
{
	int edgeThresh = 1;
	int lowThreshold;
	int const max_lowThreshold = 100;
	int ratio = 3;
	int kernel_size = 3;

	Mat dst, detected_edges;
	dst.create(img.size(), img.type());

	/// Reduce noise with a kernel 3x3
	blur(img, detected_edges, Size(3, 3));
	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);
	img.copyTo(dst, detected_edges);
	imshow(window_name, dst);
}
