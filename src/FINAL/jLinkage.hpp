#pragma once

#include <Eigen/Dense>
#include <vector>

struct Model;

struct DistanceWith {
  Model *vec;
  int distance;
};

struct Model {
  Eigen::VectorXi vec_bool;
  std::vector<DistanceWith> distanceWith;
  int mergeNumber = 1;
};

std::vector<Eigen::VectorXi>
get_bool_transform_2D(std::array<Eigen::MatrixXd, 2> keypoints_vector);

std::vector<Model> merge_model(std::vector<Eigen::VectorXi> bool_model);