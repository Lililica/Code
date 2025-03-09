#pragma once

#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>

class ImgLoader {
private:
  cv::Mat img;
  cv::Mat descriptors;
  std::vector<cv::KeyPoint> keypoints;
  std::vector<std::vector<cv::DMatch>> matches;

public:
  void load_path_image(const char *path);
  void detect_keypoints();
  void match_keypoints();
  std::array<Eigen::MatrixXd, 2> get_keypoints();
  cv::Mat get_img() { return img; };
};
