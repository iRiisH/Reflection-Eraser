# Reflection-Eraser

This project is a C++ implementation of Xue et al. approach for reflection removal ([A Computational Approach for Obstruction-Free Photography](https://sites.google.com/site/obstructionfreephotography/) using OpenCV.

Several slight changes are made, particularly in the initialization process, illustrated below.

The algorithm identifies the sparse optical flow on edges, then extract its two main "components" - corresponding respectively to the motion of the background and to the motion of the reflection - using RANSAC algorithm. Both sparse motion fields are then interpolated to dense motion fields using Delaunay triangulation + Lagrange 2D interpolation.

![illustration of the initialization process](https://github.com/iRiisH/Reflection-Eraser/blob/master/illustr.png)

Once all motion fields have been identified, the algorithm superposes all images warped according to these motions and then takes the minimum-value pixel to represent the background. The result of the initialization is shown above, and can be refined through an optimization algorithm described in the original article.
