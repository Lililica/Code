#include "jLinkage.hpp"
#include <iostream>
#include <random>

#include "linear_transform.hpp"

/**
 * @brief Generates a boolean transformation matrix for 2D keypoints.
 *
 * This function takes a vector of 2D keypoints and generates a boolean matrix
 * indicating whether the transformed keypoints are within a certain distance
 * threshold from the original keypoints.
 *
 * @param keypoints_vector An array of two Eigen::MatrixXd objects representing
 *                         the 2D keypoints.
 * @return A vector of Eigen::VectorXi objects representing the boolean
 *         transformation matrix.
 */
std::vector<Eigen::VectorXi>
get_bool_transform_2D(std::array<Eigen::MatrixXd, 2> keypoints_vector) {
  std::cout << "get_bool_transform :" << std::endl;

  // Initialize random number generator
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dis(0, keypoints_vector[0].rows() - 1);

  int const nombre_model = 1000;
  int const seuil_distance = 40;

  std::vector<Eigen::VectorXi> boolMatrix(nombre_model);

  // Generate boolean transformation matrices
  for (int model{0}; model < nombre_model; model++) {
    bool null_detector = true;
    Eigen::MatrixXd T;
    Eigen::VectorXi boolVector(keypoints_vector[0].rows());
    // Generate a random transformation matrix
    while (null_detector) {
      Eigen::MatrixXd P1(4, 3);
      Eigen::MatrixXd P2(4, 3);

      for (int i = 0; i < 4; i++) {
        int random_index = dis(gen);

        P1.row(i) = keypoints_vector[0].row(random_index);
        P2.row(i) = keypoints_vector[1].row(random_index);

        P1.row(i)[2] = 1;
        P2.row(i)[2] = 1;
      }

      T = isometric_leastsquare_transform_v3(P1, P2);

      // Check if the transformation is not null
      // If the transformation is null, generate a new one
      int compteur = 0;
      for (int i = 0; i < keypoints_vector[0].rows(); i++) {
        Eigen::Vector3d P1_ = T * keypoints_vector[0].row(i).transpose();
        Eigen::Vector3d P2_ = keypoints_vector[1].row(i);
        if ((P1_ - P2_).norm() < seuil_distance) {
          compteur++;
        }
      }
      if (compteur > 3) {
        null_detector = false;
      }
    }
    // add the boolean transformation to the vector
    for (int i = 0; i < keypoints_vector[0].rows(); i++) {
      Eigen::Vector3d P1_ = T * keypoints_vector[0].row(i).transpose();
      Eigen::Vector3d P2_ = keypoints_vector[1].row(i);
      boolVector(i) = (P1_ - P2_).norm() < seuil_distance;
    }
    boolMatrix[model] = boolVector;
    if (model % 1000 == 0)
      std::cout << "model = " << model << std::endl;
  }

  return boolMatrix;
}

std::vector<Model> merge_model(std::vector<Eigen::VectorXi> bool_model) {

  std::vector<Model> ModelList;

  // Create a list of models
  for (auto &vecBool : bool_model) {
    ModelList.push_back({vecBool, {}});
  }

  // Create a list of distances between models
  struct MaxPoint {
    Model *mod1 = nullptr;
    Model *mod2 = nullptr;
    int value = 0;
  };

  auto comparator = [](const MaxPoint &a, const MaxPoint &b) {
    return a.value < b.value;
  };
  std::priority_queue<MaxPoint, std::vector<MaxPoint>, decltype(comparator)>
      maxHeap(comparator);

  // Calculate the distance between each model
  for (size_t i = 0; i < ModelList.size(); ++i) {
    for (size_t j = i + 1; j < ModelList.size(); ++j) {
      int distance = ModelList[i].vec_bool.transpose() * ModelList[j].vec_bool;
      if (distance > 0) {
        maxHeap.push({&ModelList[i], &ModelList[j], distance});
      }
    }
  }

  int const nbrTris = 300;

  // Merge models
  for (int N{0}; N < nbrTris && !maxHeap.empty(); ++N) {
    std::cout << "N = " << N << std::endl;
    MaxPoint maxPoint = maxHeap.top();
    maxHeap.pop();

    Model *vec1 = maxPoint.mod1;
    Model *vec2 = maxPoint.mod2;

    if (vec1 == nullptr || vec2 == nullptr || vec2->vec_bool.sum() == 0) {
      continue;
    }

    // Merge boolean vectors
    vec1->vec_bool = (vec1->vec_bool.array() + vec2->vec_bool.array()).min(1);

    // Update merge count and clear vec2
    vec1->mergeNumber += vec2->mergeNumber;
    vec2->vec_bool = Eigen::VectorXi::Zero(vec2->vec_bool.size());

    // Recalculate distances for vec1
    for (auto &model : ModelList) {
      if (&model == vec1 || model.vec_bool.sum() == 0) {
        continue;
      }
      int distance = vec1->vec_bool.transpose() * model.vec_bool;
      if (distance > 0) {
        maxHeap.push({vec1, &model, distance});
      }
    }
  }

  // Remove empty models
  ModelList.erase(std::remove_if(ModelList.begin(), ModelList.end(),
                                 [](const Model &model) {
                                   return model.vec_bool.sum() == 0;
                                 }),
                  ModelList.end());

  return ModelList;
}
