 #pragma once
 
void nearestNeighbourWeightedInterpolation(Mat& img);
void testInterpolation();
void interpolateMotionField(vector<vector<Point2i>> &v);
void draw_subdiv(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color);
void locate_point(Mat& img, Subdiv2D& subdiv, Point2f fp, Scalar active_color);
void draw_subdiv_point(Mat& img, Point2f fp, Scalar color);
Subdiv2D createDelaunayTriangulation(vector<vector<Point2i>> v);
void interpolateMotionField2(vector<vector<Point2i>> &v);
float interpolatedValue(Subdiv2D& subdiv, Point2f p, const Mat& img);
float value(vector<Point2f> l, vector<float> vals, Point2f p);
float dist(Point2f a, Point2f b);
