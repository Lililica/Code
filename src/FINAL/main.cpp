#include "ImgLoader.hpp"
#include "jLinkage.hpp"
#include "linear_transform.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

int main() {
  std::cout << "Program Started" << std::endl;

  ImgLoader img_loader;
  img_loader.load_path_image("../src/FINAL/image/handspinner.png");
  img_loader.detect_keypoints();
  img_loader.match_keypoints();

  std::array<Eigen::MatrixXd, 2> keypoints = img_loader.get_keypoints();

  std::vector<Eigen::VectorXi> bool_transform =
      get_bool_transform_2D(keypoints);

  std::vector<Model> merged = merge_model(bool_transform);

  Model max;
  for (auto &model : merged) {
    if (model.mergeNumber > max.mergeNumber) {
      max = model;
    }
  }
  int nbrPoints = 0;
  for (int i = 0; i < max.vec_bool.size(); i++) {
    if (max.vec_bool(i) == 1) {
      nbrPoints++;
    }
  }

  Eigen::MatrixXd P1(nbrPoints, 3);
  Eigen::MatrixXd P2(nbrPoints, 3);

  std::cout << "nbrPoints : " << nbrPoints << std::endl;

  int compteur = 0;
  for (int i = 0; i < keypoints[0].rows(); i++) {
    if (max.vec_bool(i) == 1) {
      P1.row(compteur) = keypoints[0].row(i);
      P2.row(compteur) = keypoints[1].row(i);
      compteur++;
    }
  }

  Eigen::MatrixXd T = affine_leastsquare_transform(P1, P2);

  std::cout << "T \n" << T << std::endl;

  cv::Mat T_opencv = (cv::Mat_<double>(2, 3) << T(0, 0), T(0, 1), T(0, 2),
                      T(1, 0), T(1, 1), T(1, 2));

  cv::Mat img_rendu = img_loader.get_img().clone();

  cv::warpAffine(img_loader.get_img(), img_rendu, T_opencv,
                 img_loader.get_img().size());

  cv::Mat newIm = 0.5 * img_loader.get_img() + 0.5 * img_rendu;

  cv::imshow("image", newIm);
  cv::waitKey();

  return 0;
}