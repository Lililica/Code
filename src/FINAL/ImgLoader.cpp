#include "ImgLoader.hpp"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

void ImgLoader::load_path_image(const char *path) {
  std::cout << "Loading path: " << path << std::endl;
  cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);

  cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);

  cv::GaussianBlur(img, img, cv::Size(7, 7), 0, 0);

  cv::imshow("Image", img);
  cv::waitKey(0);

  this->img = img;
  return;
}

void ImgLoader::detect_keypoints() {

  if (img.empty()) {
    std::cerr << "Error: Image not loaded, use load_path_image before "
                 "detecting keypoints"
              << std::endl;
    return;
  }

  // Detect keypoints
  keypoints.clear();
  descriptors.release();
  cv::Ptr<cv::Feature2D> detector = cv::BRISK::create();
  detector->detectAndCompute(this->img, cv::Mat(), keypoints, descriptors);

  // Draw keypoints
  cv::Mat output_image;
  cv::drawKeypoints(img, keypoints, output_image, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::imshow("image", output_image);
  cv::waitKey(0);
}

void ImgLoader::match_keypoints() {

  if (keypoints.empty()) {
    std::cerr << "Error: No keypoints detected, use detect_keypoints before "
                 "matching keypoints"
              << std::endl;
    return;
  }

  // compute matches
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::BFMatcher::create(cv::NORM_HAMMING2, false);
  matcher->knnMatch(
      descriptors, descriptors, matches,
      5); // last param is the max number of correspondance by keypoint

  // filter matches
  int const marge_erreur = 50;

  std::vector<std::vector<cv::DMatch>> good_matches;
  for (unsigned int i = 0; i < matches.size(); ++i) {
    std::vector<cv::DMatch> good_matches_i;
    for (const auto &match : matches[i]) {
      if (match.distance != 0 && match.distance <= 100) {
        double distance = cv::norm(keypoints[match.queryIdx].pt -
                                   keypoints[match.trainIdx].pt);
        if (distance > marge_erreur) {
          good_matches_i.push_back(match);
        }
      }
    }
    good_matches.push_back(good_matches_i);
  }

  matches.clear();
  matches = good_matches;

  // sort matches by distance
  for (unsigned int i = 0; i < matches.size(); ++i)
    std::sort(matches[i].begin(), matches[i].end());
  return;
}

std::array<Eigen::MatrixXd, 2> ImgLoader::get_keypoints() {
  if (matches.empty()) {
    std::cerr << "Error: No matches detected, use match_keypoints before "
                 "extract keypoints"
              << std::endl;
    return std::array<Eigen::MatrixXd, 2>();
  }

  std::array<Eigen::MatrixXd, 2> keypoints_vector = {
      Eigen::MatrixXd(matches.size() * 5, 3),
      Eigen::MatrixXd(matches.size() * 5, 3)};
  Eigen::MatrixX3d start_matrix(matches.size() * 5, 3);
  Eigen::MatrixX3d end_matrix(matches.size() * 5, 3);

  int compteur = 0;
  for (unsigned int i = 0; i < matches.size(); ++i) {
    if (matches[i].size() < 1) {
      continue;
    }

    for (const auto &match : matches[i]) {

      start_matrix(compteur, 0) = keypoints[match.queryIdx].pt.x;
      start_matrix(compteur, 1) = keypoints[match.queryIdx].pt.y;
      start_matrix(compteur, 2) = 0;

      keypoints_vector[0].row(compteur) = start_matrix.row(compteur);

      end_matrix(compteur, 0) = keypoints[match.trainIdx].pt.x;
      end_matrix(compteur, 1) = keypoints[match.trainIdx].pt.y;
      end_matrix(compteur, 2) = 0;

      keypoints_vector[1].row(compteur) = end_matrix.row(compteur);

      compteur++;
    }
  }

  keypoints_vector[0].conservativeResize(compteur, 3);
  keypoints_vector[1].conservativeResize(compteur, 3);

  return keypoints_vector;
}
