#include <cassert>
#include <cmath>
#include <iomanip>
#include <limits>
#include <stdexcept>

#include <Eigen/Dense>

#include "linear_transform.hpp"

// Eigen::MatrixXd affine_leastsquare_transform_2d(const Eigen::MatrixXd &P1,
// const Eigen::MatrixXd &P2)
// {
//     // check that the matrices have the same size
//     assert( (P1.rows() == P2.rows()) && "P1 and P2 must have the same number
//     of rows" ); assert( (P1.cols() == P2.cols()) && "P1 and P2 must have the
//     same number of cols" );

//     // build matrix M and vector v
//     Eigen::MatrixXd M(2*P1.rows(), 6);
//     Eigen::VectorXd v(2*P1.rows());
//     M.setZero();
//     for(unsigned int i=0; i<P1.rows(); ++i){
//         M(2*i, 0) = P1(i, 0);
//         M(2*i, 1) = P1(i, 1);
//         M(2*i, 4) = 1.0;
//         M((2*i)+1, 2) = P1(i, 0);
//         M((2*i)+1, 3) = P1(i, 1);
//         M((2*i)+1, 5) = 1.0;

//         v(2*i)     = P2(i, 0);
//         v((2*i)+1) = P2(i, 1);
//     }
//     // std::cout << "M \n" << M << std::endl;

//     // least square solution
//     Eigen::VectorXd u = (M.transpose() * M).inverse() * M.transpose() * v;

//     // build the affine transform matrix
//     Eigen::MatrixXd T(3, 3);
//     T << u(0), u(1), u(4),
//          u(2), u(3), u(5),
//          0.0,  0.0,  1.0;

//     return T;
// }

void clean_zeros(Eigen::MatrixXd &M, const double epsilon) {
  for (int i = 0; i < M.rows(); ++i)
    for (int j = 0; j < M.cols(); ++j)
      if (std::abs(M(i, j)) < epsilon)
        M(i, j) = 0.0;
}

Eigen::MatrixXd affine_leastsquare_transform(const Eigen::MatrixXd &P1,
                                             const Eigen::MatrixXd &P2) {
  // check that the matrices have the same size
  assert((P1.rows() == P2.rows()) &&
         "P1 and P2 must have the same number of rows");
  assert((P1.cols() == P2.cols()) &&
         "P1 and P2 must have the same number of cols");

  // space dimension in Pn is n-1
  const int dimension = P1.cols() - 1;

  // build matrix M and vector v
  Eigen::MatrixXd M(dimension * P1.rows(), dimension * (dimension + 1));
  Eigen::VectorXd v(dimension * P1.rows());
  M.setZero();
  // for each point correspondance
  for (unsigned int i = 0; i < P1.rows(); ++i) {
    // for each dimension of a point
    for (int j = 0; j < dimension; ++j) {
      // add all points componenents
      for (int k = 0; k < dimension; ++k)
        M((dimension * i) + j, (dimension * j) + k) = P1(i, k);

      // add the translation component
      M((dimension * i) + j, (dimension * dimension) + j) = 1.0;

      // compute vector v
      v((dimension * i) + j) = P2(i, j);
    }
  }
  // std::cout << "M \n" << M << std::endl;

  // least square solution
  Eigen::VectorXd u = (M.transpose() * M).inverse() * M.transpose() * v;

  // build the affine transform matrix
  Eigen::MatrixXd T(dimension + 1, dimension + 1);
  T.setIdentity();
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j)
      T(i, j) = u((dimension * i) + j);

    T(i, dimension) = u((dimension * dimension) + i);
  }

  // std::cout << "T \n" << T << std::endl;

  return T;
}

Eigen::MatrixXd isometric_leastsquare_transform_v1(const Eigen::MatrixXd &P1,
                                                   const Eigen::MatrixXd &P2) {
  // first get the affine transform
  Eigen::MatrixXd M = affine_leastsquare_transform(P1, P2);

  // then extract the translation, rotation and reflexion from the affine
  // transform
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      M.topLeftCorner(M.rows() - 1, M.cols() - 1),
      Eigen::ComputeFullU | Eigen::ComputeFullV);
  M.topLeftCorner(M.rows() - 1, M.cols() - 1) =
      svd.matrixU() * svd.matrixV().transpose();
  // clean_zeros(M);

  return M;
}

Eigen::MatrixXd isometric_leastsquare_transform_v2(const Eigen::MatrixXd &P1,
                                                   const Eigen::MatrixXd &P2) {
  // first get the affine transform
  Eigen::MatrixXd M = affine_leastsquare_transform(P1, P2);

  // the same with QR decomposition
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
  Eigen::MatrixXd Q = qr.householderQ();
  Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

  // remove affine part
  for (unsigned int i = 0; i < R.rows() - 1; ++i)
    for (unsigned int j = i + 1; j < R.cols() - 1; ++j)
      R(i, j) = 0.0;

  // std::cout << "R after \n" << R << std::endl << std::endl;
  M = Q * R;

  return M;
}

// get that one
Eigen::MatrixXd isometric_leastsquare_transform_v3(const Eigen::MatrixXd &P1,
                                                   const Eigen::MatrixXd &P2) {
  // check that the matrices have the same size
  assert((P1.rows() == P2.rows()) &&
         "P1 and P2 must have the same number of rows");
  assert((P1.cols() == P2.cols()) &&
         "P1 and P2 must have the same number of cols");

  // remove the homogeneous component
  Eigen::MatrixXd P1_ = P1.topLeftCorner(P1.rows(), P1.cols() - 1);
  Eigen::MatrixXd P2_ = P2.topLeftCorner(P2.rows(), P2.cols() - 1);
  // std::cout << std::setprecision(2) << "\nP1\n" << P1 << std::endl;
  // std::cout << std::setprecision(2) << "\nP2\n" << P2 << std::endl;

  // mean of the data
  Eigen::VectorXd mean_1 = P1_.colwise().sum() / (double)P1_.rows();
  Eigen::VectorXd mean_2 = P2_.colwise().sum() / (double)P2_.rows();
  // std::cout << "\nmean_1 : " << mean_1.transpose() << std::endl;
  // std::cout << "\nmean_2 : " << mean_2.transpose() << std::endl;

  // center the data
  P1_ = P1_ - Eigen::VectorXd::Ones(P1.rows()) * mean_1.transpose();
  P2_ = P2_ - Eigen::VectorXd::Ones(P2.rows()) * mean_2.transpose();
  // std::cout << std::setprecision(2) << "\nP1_centered\n" << P1_ << std::endl;
  // std::cout << std::setprecision(2) << "\nP2_centered\n" << P2_ << std::endl;

  // compute the covariance matrix
  Eigen::MatrixXd H = (P1_.transpose() * P2_);
  // std::cout  << std::setprecision(2) << "\nH\n" << H << std::endl;

  // compute the SVD of H
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU |
                                               Eigen::ComputeFullV);
  Eigen::MatrixXd R = svd.matrixU() * (svd.matrixV().transpose());
  // std::cout  << std::setprecision(2) << "\nR\n" << R << std::endl;

  // Eigen::MatrixXd D = svd.singularValues().asDiagonal().toDenseMatrix();
  //  std::cout  << "\nD\n" << D << std::endl;
  //  std::cout  << "\nSVD zero\n" << H - svd.matrixU() * D *
  //  svd.matrixV().transpose() << std::endl << std::endl;

  // strange debug : if the rotation matrix is not orthogonal, transpose it ???
  if ((R * P1_.transpose() - P2_.transpose()).norm() > 1.e-6) {
    // std::cout << "**************************************" << std::endl;
    // std::cout  << std::setprecision(2) << "\nold R\n" << R << std::endl;
    // R = R.transpose();  // bug !!!!!!!!!!!
    R.transposeInPlace();
    // std::cout << "svd debug" << std::endl;  /// \todo clarify this
    //  std::cout  << std::setprecision(2) << "\nnew R\n" << R << std::endl;
  }
  // std::cout << "P1_ : \n" << P1_ << std::endl;
  // std::cout << "P2_ : \n" << P2_ << std::endl;
  // std::cout << "R on P1 : \n" << (R*P1_.transpose()).transpose() <<
  // std::endl; std::cout << "R on P1 - P2 : \n" << (R*P1_.transpose() -
  // P2_.transpose()).transpose() << std::endl;

  // if(R.determinant() < 0){
  //     std::cout << "reflexion" << std::endl;
  // }

  // build the isometric transform matrix
  Eigen::MatrixXd M = Eigen::MatrixXd::Identity(P1.cols(), P1.cols());
  M.topLeftCorner(P1.cols() - 1, P1.cols() - 1) = R;
  M.topRightCorner(P1.cols() - 1, 1) = mean_2 - R * mean_1;

  clean_zeros(M);
  //   std::cout << std::setprecision(2) << "\nM\n" << M << std::endl;

  return M;
}

Eigen::MatrixXd homography_leastsquare_transform_P2(const Eigen::MatrixXd &P1,
                                                    const Eigen::MatrixXd &P2) {
  // check that the matrices have the same size
  assert((P1.rows() == P2.rows()) &&
         "P1 and P2 must have the same number of rows");
  assert((P1.cols() == P2.cols()) &&
         "P1 and P2 must have the same number of cols");
  assert((P1.cols() == 3) && "only defined in P2");

  if (P1.rows() < 4)
    throw std::length_error(
        "2D homographies requires at least 4 point correspondances");

  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(P1.rows() * 2, 9);
  for (unsigned int i = 0; i < P1.rows(); ++i) {

    // first line
    M(2 * i, 3) = -P2(i, 2) * P1(i, 0);
    M(2 * i, 4) = -P2(i, 2) * P1(i, 1);
    M(2 * i, 5) = -P2(i, 2) * P1(i, 2);

    M(2 * i, 6) = P2(i, 1) * P1(i, 0);
    M(2 * i, 7) = P2(i, 1) * P1(i, 1);
    M(2 * i, 8) = P2(i, 1) * P1(i, 2);

    // second line
    M(2 * i + 1, 0) = P2(i, 2) * P1(i, 0);
    M(2 * i + 1, 1) = P2(i, 2) * P1(i, 1);
    M(2 * i + 1, 2) = P2(i, 2) * P1(i, 2);

    M(2 * i + 1, 6) = -P2(i, 0) * P1(i, 0);
    M(2 * i + 1, 7) = -P2(i, 0) * P1(i, 1);
    M(2 * i + 1, 8) = -P2(i, 0) * P1(i, 2);
  }

  // check rank and determinant
  Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(M);
  if (lu_decomp.rank() < 8)
    throw std::domain_error("2D homography: data should be at least rank 9:"
                            "you may have aligned points");
  // if(std::abs(lu_decomp.determinant()) < 1.0e-8) throw std::domain_error("2D
  // homography: data should non zero determinant: you may have aligned
  // points");

  // solve (least square)
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU |
                                               Eigen::ComputeFullV);
  Eigen::MatrixXd h = svd.matrixV().col(M.cols() - 1);
  // std::cout << "\nh " << h.transpose() << std::endl;

  // convert h to matrix
  Eigen::MatrixXd H(3, 3);
  H << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);

  // H /= H(2,2);
  // std::cout << "\nH\n" << H << std::endl;

  return H;
}

Eigen::MatrixXd homography_leastsquare_transform_P3(const Eigen::MatrixXd &P1,
                                                    const Eigen::MatrixXd &P2) {
  // check that the matrices have the same size
  assert((P1.rows() == P2.rows()) &&
         "P1 and P2 must have the same number of rows");
  assert((P1.cols() == P2.cols()) &&
         "P1 and P2 must have the same number of cols");
  assert((P1.cols() == 4) && "only defined in P3");

  if (P1.rows() < 5)
    throw std::length_error(
        "3D homographies requires at least 5 point correspondances");

  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(P1.rows() * 6, 16);
  for (unsigned int i = 0; i < P1.rows(); ++i) {

    // first line
    M(6 * i, 4) = P2(i, 0) * P1(i, 0);
    M(6 * i, 5) = P2(i, 0) * P1(i, 1);
    M(6 * i, 6) = P2(i, 0) * P1(i, 2);
    M(6 * i, 7) = P2(i, 0) * P1(i, 3);

    M(6 * i, 0) = -P2(i, 1) * P1(i, 0);
    M(6 * i, 1) = -P2(i, 1) * P1(i, 1);
    M(6 * i, 2) = -P2(i, 1) * P1(i, 2);
    M(6 * i, 3) = -P2(i, 1) * P1(i, 3);

    // second line
    M(6 * i + 1, 8) = P2(i, 0) * P1(i, 0);
    M(6 * i + 1, 9) = P2(i, 0) * P1(i, 1);
    M(6 * i + 1, 10) = P2(i, 0) * P1(i, 2);
    M(6 * i + 1, 11) = P2(i, 0) * P1(i, 3);

    M(6 * i + 1, 0) = -P2(i, 2) * P1(i, 0);
    M(6 * i + 1, 1) = -P2(i, 2) * P1(i, 1);
    M(6 * i + 1, 2) = -P2(i, 2) * P1(i, 2);
    M(6 * i + 1, 3) = -P2(i, 2) * P1(i, 3);

    // third line
    M(6 * i + 2, 12) = P2(i, 0) * P1(i, 0);
    M(6 * i + 2, 13) = P2(i, 0) * P1(i, 1);
    M(6 * i + 2, 14) = P2(i, 0) * P1(i, 2);
    M(6 * i + 2, 15) = P2(i, 0) * P1(i, 3);

    M(6 * i + 2, 0) = -P2(i, 3) * P1(i, 0);
    M(6 * i + 2, 1) = -P2(i, 3) * P1(i, 1);
    M(6 * i + 2, 2) = -P2(i, 3) * P1(i, 2);
    M(6 * i + 2, 3) = -P2(i, 3) * P1(i, 3);

    // fourth line
    M(6 * i + 3, 8) = P2(i, 1) * P1(i, 0);
    M(6 * i + 3, 9) = P2(i, 1) * P1(i, 1);
    M(6 * i + 3, 10) = P2(i, 1) * P1(i, 2);
    M(6 * i + 3, 11) = P2(i, 1) * P1(i, 3);

    M(6 * i + 3, 4) = -P2(i, 2) * P1(i, 0);
    M(6 * i + 3, 5) = -P2(i, 2) * P1(i, 1);
    M(6 * i + 3, 6) = -P2(i, 2) * P1(i, 2);
    M(6 * i + 3, 7) = -P2(i, 2) * P1(i, 3);

    // fifth line
    M(6 * i + 4, 12) = P2(i, 1) * P1(i, 0);
    M(6 * i + 4, 13) = P2(i, 1) * P1(i, 1);
    M(6 * i + 4, 14) = P2(i, 1) * P1(i, 2);
    M(6 * i + 4, 15) = P2(i, 1) * P1(i, 3);

    M(6 * i + 4, 4) = -P2(i, 3) * P1(i, 0);
    M(6 * i + 4, 5) = -P2(i, 3) * P1(i, 1);
    M(6 * i + 4, 6) = -P2(i, 3) * P1(i, 2);
    M(6 * i + 4, 7) = -P2(i, 3) * P1(i, 3);

    // sixth line
    M(6 * i + 5, 12) = P2(i, 2) * P1(i, 0);
    M(6 * i + 5, 13) = P2(i, 2) * P1(i, 1);
    M(6 * i + 5, 14) = P2(i, 2) * P1(i, 2);
    M(6 * i + 5, 15) = P2(i, 2) * P1(i, 3);

    M(6 * i + 5, 8) = -P2(i, 3) * P1(i, 0);
    M(6 * i + 5, 9) = -P2(i, 3) * P1(i, 1);
    M(6 * i + 5, 10) = -P2(i, 3) * P1(i, 2);
    M(6 * i + 5, 11) = -P2(i, 3) * P1(i, 3);
  }

  // std::cout << "\nP1\n" << P1 << std::endl;
  // std::cout << "\nP1.cols = " << P1.cols() << std::endl;
  // std::cout << "\nP1.rows = " << P1.rows() << std::endl;
  // std::cout << "\nP2\n" << P2 << std::endl;
  // std::cout << "\nM\n" << M << std::endl;
  // Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(M);
  // std::cout << "\nrank(M) = " << lu_decomp.rank() << std::endl;

  // check rank and determinant
  Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(M);
  // std::cout << "rank = " <<lu_decomp.rank() << std::endl;
  if (lu_decomp.rank() < 12)
    throw std::domain_error("3D homography: data should be at least rank xxx: "
                            "you may have aligned points");
  // if(std::abs(lu_decomp.determinant()) < 1.0e-8) throw std::domain_error("3D
  // homography: data should non zero determinant: you may have aligned
  // points");

  // solve (least square)
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU |
                                               Eigen::ComputeFullV);
  Eigen::MatrixXd h = svd.matrixV().col(M.cols() - 1);

  // std::cout << "\nV\n" << svd.matrixV() << std::endl;
  // std::cout << "\nh\n" << h.transpose() << std::endl;

  // convert h to matrix
  Eigen::MatrixXd H(4, 4);
  H << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8), h(9), h(10), h(11),
      h(12), h(13), h(14), h(15);

  // H /= H(3,3);
  // clean_zeros(H);
  // std::cout << "\nH\n" << H << std::endl;

  return H;
}

// Eigen::MatrixXd homography_leastsquare_transform_v2(const Eigen::MatrixXd
// &P1, const Eigen::MatrixXd &P2)
// {
//     // check that the matrices have the same size
//     assert( (P1.rows() == P2.rows()) && "P1 and P2 must have the same number
//     of rows" ); assert( (P1.cols() == P2.cols()) && "P1 and P2 must have the
//     same number of cols" );

//     Eigen::MatrixXd P1_inv = (P1.transpose()*P1).inverse() * P1.transpose();
//     Eigen::MatrixXd H = P2.transpose() * P1_inv.transpose();

//     return H;
// }

double symmetric_transfer_error(const Eigen::MatrixXd &M,
                                const Eigen::MatrixXd &P1,
                                const Eigen::MatrixXd &P2) {
  // check that the matrices have the same size
  assert((P1.rows() == P2.rows()) &&
         "P1 and P2 must have the same number of rows");
  assert((P1.cols() == P2.cols()) &&
         "P1 and P2 must have the same number of cols");

  // transform P1
  Eigen::MatrixXd P2_hat = M * P1.transpose();

  // convert from Arguesian to Eucliedan
  for (unsigned int j = 0; j < P2_hat.cols(); ++j)
    if (std::abs(P2_hat(P2_hat.rows() - 1, j)) >
        std::numeric_limits<double>::epsilon())
      P2_hat.col(j) /= P2_hat(P2_hat.rows() - 1, j);

  // transform P2
  Eigen::MatrixXd P1_hat = M.inverse() * P2.transpose();
  for (unsigned int j = 0; j < P1_hat.cols(); ++j)
    if (std::abs(P1_hat(P1_hat.rows() - 1, j)) >
        std::numeric_limits<double>::epsilon())
      P1_hat.col(j) /= P1_hat(P2_hat.rows() - 1, j);

  // compute errors
  double error_12 =
      (P2 - P2_hat.transpose()).topLeftCorner(P1.rows(), P1.cols() - 1).norm();
  double error_21 =
      (P1 - P1_hat.transpose()).topLeftCorner(P1.rows(), P1.cols() - 1).norm();

  // combine both errors
  return error_12 + error_21;
}
