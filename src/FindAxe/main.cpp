#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#define ASSETS_PATH "../src/FindAxe/assets/"

int main() {
  std::cout << "Find Axe : " << std::endl;

  cv::Mat image;
  // Load the image
  image = cv::imread(ASSETS_PATH "test.jpeg");

  if (image.empty()) {
    std::cerr << "Could not open or find the image!" << std::endl;
    return -1;
  }

  cv::imshow("Original Image", image);
  cv::waitKey(0);

  //   Generate a random affine matrix transformation

  double sx = 1;
  double sy = 1;
  double teta = M_PI / 4.2;
  double tx = 200;
  double ty = 200;

  Eigen::MatrixXd affineMatrix = Eigen::MatrixXd::Identity(3, 3);
  //   C1
  affineMatrix(0, 0) = sx * cos(teta);
  affineMatrix(1, 0) = sy * sin(teta);
  affineMatrix(2, 0) = 0;
  //   C2
  affineMatrix(0, 1) = -sx * sin(teta);
  affineMatrix(1, 1) = sy * cos(teta);
  affineMatrix(2, 1) = 0;
  //   C3
  affineMatrix(0, 2) = tx;
  affineMatrix(1, 2) = ty;
  affineMatrix(2, 2) = 1;

  //   Print the affine matrix
  std::cout
      << "_________________________________________________________________"
      << std::endl;
  std::cout << "Affine Matrix: " << std::endl;
  std::cout << affineMatrix << std::endl;

  // Convert the transformation matrix to OpenCV format
  cv::Mat T_opencv =
      (cv::Mat_<double>(2, 3) << affineMatrix(0, 0), affineMatrix(0, 1),
       affineMatrix(0, 2), affineMatrix(1, 0), affineMatrix(1, 1),
       affineMatrix(1, 2));

  // Apply the affine transformation to the image
  cv::Mat imageModif;

  cv::warpAffine(image, imageModif, T_opencv, image.size());

  cv::imshow("Transformed Image", image * 0.5 + imageModif * 0.5);
  cv::waitKey(0);

  // _____________________________________________________
  // Maths section

  // Avoir les valeurs propres de la matrice affine
  Eigen::EigenSolver<Eigen::MatrixXd> es(affineMatrix.block<3, 3>(0, 0));
  Eigen::VectorXd eigenvalues = es.eigenvalues().real();

  std::cout
      << "_________________________________________________________________"
      << std::endl;
  std::cout << "Valeurs propres  de la matrix global: " << std::endl;
  std::cout << eigenvalues << std::endl;

  // Avoir les vecteurs propres de la matrice affine
  Eigen::MatrixXd eigenvectors = es.eigenvectors().real();
  std::cout
      << "_________________________________________________________________"
      << std::endl;
  std::cout << "Vecteurs propres de la matrix global : " << std::endl;
  std::cout << eigenvectors << std::endl;

  //   Get the QR decomposition of the affine matrix
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(affineMatrix);
  Eigen::MatrixXd Q = qr.householderQ();
  Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

  //   Print the Q and R matrices

  std::cout
      << "_________________________________________________________________"
      << std::endl;
  std::cout << "Q Matrix: " << std::endl;
  std::cout << Q << std::endl;

  // valeurs propres de la matrice Q
  Eigen::EigenSolver<Eigen::MatrixXd> esQ(Q);
  Eigen::VectorXd eigenvaluesQ = esQ.eigenvalues().real();
  std::cout
      << "_________________________________________________________________"
      << std::endl;
  std::cout << "Valeurs propres de la matrix Q : " << std::endl;
  std::cout << eigenvaluesQ << std::endl;

  // vecteurs propres de la matrice Q
  Eigen::MatrixXd eigenvectorsQ = esQ.eigenvectors().real();

  std::cout
      << "_________________________________________________________________"
      << std::endl;
  std::cout << "Vecteurs propres de la matrix Q : " << std::endl;
  std::cout << eigenvectorsQ << std::endl;

  std::cout
      << "_________________________________________________________________"
      << std::endl;
  std::cout << "R Matrix: " << std::endl;
  std::cout << R << std::endl;

  // valeurs propres de la matrice Q
  Eigen::EigenSolver<Eigen::MatrixXd> esR(R);
  Eigen::VectorXd eigenvaluesR = esR.eigenvalues().real();
  std::cout
      << "_________________________________________________________________"
      << std::endl;
  std::cout << "Valeurs propres de la matrix R : " << std::endl;
  std::cout << eigenvaluesR << std::endl;

  // vecteurs propres de la matrice Q
  Eigen::MatrixXd eigenvectorsR = esR.eigenvectors().real();

  std::cout
      << "_________________________________________________________________"
      << std::endl;
  std::cout << "Vecteurs propres de la matrix R : " << std::endl;
  std::cout << eigenvectorsR << std::endl;

  //   //   Get the determinant of the Q matrix
  //   double detQ = Q.determinant();

  //   if (detQ < 0) {
  //     std::cout
  //         <<
  //         "_________________________________________________________________"
  //         << std::endl;
  //     std::cout << "Inversion du sens" << std::endl;

  //     // decompose la matrice Q en Q2 et R2
  //     Eigen::HouseholderQR<Eigen::MatrixXd> qr2(Q);
  //     Eigen::MatrixXd Q2 = qr2.householderQ();
  //     Eigen::MatrixXd R2 = qr2.matrixQR().triangularView<Eigen::Upper>();
  //     //   Print the Q2 and R2 matrices

  //     std::cout << "Q2 Matrix: " << std::endl;
  //     std::cout << Q2 << std::endl;
  //     std::cout << "R2 Matrix: " << std::endl;
  //     std::cout << R2 << std::endl;
  //   }

  return 0;
}