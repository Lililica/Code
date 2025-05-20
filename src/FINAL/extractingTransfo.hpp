#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using namespace std;
using namespace Eigen;

// Function to generate a random rotation matrix using QR decomposition in 3D
Eigen::Matrix3d randomRotationMatrix();
void decomposeHomography(const Matrix4d &H, Matrix3d &R, Vector3d &t,
                         double &scale);

void rotationMatrixToAxisAngle(const Matrix3d &R, Vector3d &axis,
                               double &angle);

void HomographyToTransform3D(Matrix4d &H);

// Function to generate a random rotation matrix using QR decomposition in 2D
void decomposeHomography2D(const Matrix3d &H, Matrix2d &R, Vector2d &t,
                           double &scale, double &angle);
void HomographyToTransform2D(Matrix3d &H);