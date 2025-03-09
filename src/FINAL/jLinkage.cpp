#include "jLinkage.hpp"
#include <iostream>
#include <random>

#include "linear_transform.hpp"

std::vector<Eigen::VectorXi>
get_bool_transform_2D(std::array<Eigen::MatrixXd, 2> keypoints_vector) {
  std::cout << "get_bool_transform :" << std::endl;

  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dis(0, keypoints_vector[0].rows() - 1);

  int const nombre_model = 10000;
  int const seuil_distance = 10;

  std::vector<Eigen::VectorXi> boolMatrix(nombre_model);

  for (int model{0}; model < nombre_model; model++) {
    bool null_detector = true;
    Eigen::MatrixXd T;
    Eigen::VectorXi boolVector(keypoints_vector[0].rows());
    while (null_detector) {
      Eigen::MatrixXd P1(4, 3);
      Eigen::MatrixXd P2(4, 3);

      for (int i = 0; i < 4; i++) {
        int random_index = dis(gen);

        P1.row(i) = keypoints_vector[0].row(random_index);
        P2.row(i) = keypoints_vector[1].row(random_index);
      }

      T = affine_leastsquare_transform(P1, P2);

      for (int i = 0; i < keypoints_vector[0].rows(); i++) {
        Eigen::Vector3d P1_ = T * keypoints_vector[0].row(i).transpose();
        Eigen::Vector3d P2_ = keypoints_vector[1].row(i);
        if ((P1_ - P2_).norm() < seuil_distance) {
          null_detector = false;
        }
      }
    }
    for (int i = 0; i < keypoints_vector[0].rows(); i++) {
      Eigen::Vector3d P1_ = T * keypoints_vector[0].row(i).transpose();
      Eigen::Vector3d P2_ = keypoints_vector[1].row(i);
      boolVector(i) = (P1_ - P2_).norm() < seuil_distance;
    }
    boolMatrix[model] = boolVector;
  }

  return boolMatrix;
}

std::vector<Model> merge_model(std::vector<Eigen::VectorXi> bool_model) {

  std::vector<Model> ModelList;

  for (auto &vecBool : bool_model) {
    ModelList.push_back({vecBool, {}});
  }

  struct MaxPoint {
    Model *mod1 = nullptr;
    Model *mod2 = nullptr;
    int value = 0;
  };

  std::vector<MaxPoint> maxList;

  for (auto &model : ModelList) {
    for (auto &model2 : ModelList) {
      if (&model == &model2) {
        continue;
      }

      model.distanceWith.push_back(
          DistanceWith{&model2, model.vec_bool.transpose() * model2.vec_bool});
      if (model.distanceWith.back().distance > 0) {
        maxList.push_back(
            {&model, &model2, model.distanceWith.back().distance});
      }
    }
  }

  if (maxList.size() == 0) {
    std::cerr << "No model found" << std::endl;
    return std::vector<Model>();
  }

  int const nbrTris = 100;

  for (int N{0}; N < nbrTris; ++N) {
    std::sort(
        maxList.begin(), maxList.end(),
        [](const MaxPoint &a, const MaxPoint &b) { return a.value < b.value; });
    Model *vec1 = maxList[maxList.size() - 1].mod1;
    Model *vec2 = maxList[maxList.size() - 1].mod2;

    for (int i = 0; i < vec1->vec_bool.size(); i++) {
      if (vec1->vec_bool(i) == 1 || vec2->vec_bool(i) == 1) {
        vec1->vec_bool(i) = 1;
      }
    }

    vec2->vec_bool = Eigen::VectorXi::Zero(vec2->vec_bool.size());
    vec2->distanceWith.clear();
    vec1->distanceWith.clear();
    vec1->mergeNumber += vec2->mergeNumber;
    maxList.pop_back();

    for (auto &model : ModelList) {
      for (int i = 0; i < model.distanceWith.size(); i++) {
        if (model.distanceWith[i].vec == vec2 ||
            model.distanceWith[i].vec == vec1) {
          model.distanceWith.erase(model.distanceWith.begin() + i);
        }
      }
    }
    for (int i = 0; i < maxList.size(); i++) {
      auto &max = maxList[i];
      if (max.mod1 == vec1 || max.mod1 == vec2 || max.mod2 == vec1 ||
          max.mod2 == vec2) {
        maxList.erase(maxList.begin() + i);
      }
    }

    for (auto &model : ModelList) {
      if (&model == vec1) {
        continue;
      }
      vec1->distanceWith.push_back(
          DistanceWith{&model, vec1->vec_bool.transpose() * model.vec_bool});
      if (vec1->distanceWith.back().distance > 0) {
        maxList.push_back({vec1, &model, model.distanceWith.back().distance});
      }
    }
  }

  return ModelList;
}