#include "ImgLoader.hpp"
#include "jLinkage.hpp"
#include "linear_transform.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

/**
 * @brief Main function to load an image, detect and match keypoints, and apply
 * an affine transformation.
 *
 * The program performs the following steps:
 * 1. Prints a message indicating the program has started.
 * 2. Loads an image using ImgLoader.
 * 3. Detects keypoints in the image.
 * 4. Matches the detected keypoints.
 * 5. Retrieves the keypoints as an array of Eigen::MatrixXd.
 * 6. Converts the keypoints to a boolean transformation vector.
 * 7. Merges the models based on the boolean transformation.
 * 8. Finds the model with the maximum number of merges.
 * 9. Counts the number of points in the model with the maximum merges.
 * 10. Prints the number of merges and points.
 * 11. Creates matrices P1 and P2 to store the keypoints.
 * 12. Fills P1 and P2 with the corresponding keypoints.
 * 13. Computes the affine transformation matrix T using least squares.
 * 14. Prints the affine transformation matrix T.
 * 15. Converts the Eigen matrix T to an OpenCV matrix.
 * 16. Clones the original image.
 * 17. Applies the affine transformation to the cloned image.
 * 18. Blends the original and transformed images.
 * 19. Displays the blended image in a window.
 * 20. Waits for a key press before exiting.
 *
 * @return int Exit status of the program.
 */
int main() {
  std::cout << "Program Started" << std::endl;

  // Load image
  ImgLoader img_loader;

  // Add image to the loader
  img_loader.load_path_image("../src/FINAL/image/handspinner.png");

  // Detect keypoints in the image
  img_loader.detect_keypoints();

  // Match keypoints between images
  img_loader.match_keypoints();

  // Retrieve keypoints from the image loader
  std::array<Eigen::MatrixXd, 2> keypoints = img_loader.get_keypoints();

  // Get boolean transformation of the keypoints
  std::vector<Eigen::VectorXi> bool_transform =
      get_bool_transform_2D(keypoints);

  // Merge models based on boolean transformation
  std::vector<Model> merged = merge_model(bool_transform);

  // Find the model with the maximum number of merges
  Model max;
  for (auto &model : merged) {
    if (model.mergeNumber > max.mergeNumber) {
      max = model;
    }
  }

  // Count the number of points in the model
  int nbrPoints = 0;
  for (int i = 0; i < max.vec_bool.size(); i++) {
    if (max.vec_bool(i) == 1) {
      nbrPoints++;
    }
  }

  std::cout << "nbrMerge : " << max.mergeNumber << std::endl;

  // Initialize matrices to store the keypoints
  Eigen::MatrixXd P1(nbrPoints, 3);
  Eigen::MatrixXd P2(nbrPoints, 3);

  std::cout << "nbrPoints : " << nbrPoints << std::endl;

  // Populate the matrices with the keypoints
  int compteur = 0;
  for (int i = 0; i < keypoints[0].rows(); i++) {
    if (max.vec_bool(i) == 1) {
      P1.row(compteur) = keypoints[0].row(i);
      P2.row(compteur) = keypoints[1].row(i);
      compteur++;
    }
  }

  // Compute the affine least squares transformation
  Eigen::MatrixXd T = isometric_leastsquare_transform_v3(P1, P2);

  std::cout << "T \n" << T << std::endl;

  // Convert the transformation matrix to OpenCV format
  cv::Mat T_opencv = (cv::Mat_<double>(2, 3) << T(0, 0), T(0, 1), T(0, 2),
                      T(1, 0), T(1, 1), T(1, 2));

  // Clone the original image for rendering
  cv::Mat img_rendu = img_loader.get_img().clone();

  // Apply the affine transformation to the image
  cv::warpAffine(img_loader.get_img(), img_rendu, T_opencv,
                 img_loader.get_img().size());

  // Blend the original and transformed images
  cv::Mat newIm = 0.5 * img_loader.get_img() + 0.5 * img_rendu;

  // Display the blended image
  cv::imshow("image", newIm);
  cv::imwrite("../src/FINAL/image/handspinnerFinal.png", newIm);
  cv::waitKey();

  // Extract rotation, scale and translation from matrix

  return 0;
}
