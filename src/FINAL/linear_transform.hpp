#pragma once

#include <Eigen/Dense>
#include <iostream>

// compute the least square affine transform from P1 to P2 in 2D
// Eigen::MatrixXd affine_leastsquare_transform_2d(const Eigen::MatrixXd &P1,
// const Eigen::MatrixXd &P2);

// compute the least square affine transform from P1 to P2 in dimension n >= 2
Eigen::MatrixXd affine_leastsquare_transform(const Eigen::MatrixXd &P1,
                                             const Eigen::MatrixXd &P2);

// compute the least square affine transform from P1 to P2 in dimension n >= 2
Eigen::MatrixXd isometric_leastsquare_transform_v1(const Eigen::MatrixXd &P1,
                                                   const Eigen::MatrixXd &P2);
Eigen::MatrixXd isometric_leastsquare_transform_v2(const Eigen::MatrixXd &P1,
                                                   const Eigen::MatrixXd &P2);
Eigen::MatrixXd isometric_leastsquare_transform_v3(const Eigen::MatrixXd &P1,
                                                   const Eigen::MatrixXd &P2);

// compute the least square homography transform from P1 to P2 in dimension n >=
// 2
Eigen::MatrixXd
homography_leastsquare_transform_P2(const Eigen::MatrixXd &P1,
                                    const Eigen::MatrixXd &P2); // only for P2
Eigen::MatrixXd
homography_leastsquare_transform_P3(const Eigen::MatrixXd &P1,
                                    const Eigen::MatrixXd &P2); // only for P2
// Eigen::MatrixXd homography_leastsquare_transform_v2(const Eigen::MatrixXd
// &P1, const Eigen::MatrixXd &P2);

// distance between M*P1 and P2 plus distance between P1 and M^-1*P2
double symmetric_transfer_error(
    const Eigen::MatrixXd &M, const Eigen::MatrixXd &P1,
    const Eigen::MatrixXd &P2); /// \todo make it really in both directions

// change super small values into zeros
void clean_zeros(Eigen::MatrixXd &M, const double epsilon = 1e-8);
