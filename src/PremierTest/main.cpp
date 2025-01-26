// Main file: main.cpp
#include "copyMove.hpp"
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

/* lililica@Mac build %
 * /Users/lililica/Documents/Tremplin_Recherche/Code/build/PremierTest
 * ../src/PremierTest/input/simple.jpg ../src/PremierTest/output/coucou.jpg */

template <typename T> void displayVec(std::vector<T> &v) {
  std::cout << "[";
  for (T element : v) {
    std::cout << element << " ,";
    if (element != v[v.size() - 1])
      std::cout << " ,";
  }
  std::cout << " ]" << std::endl;
}

void processImage(const std::string &imagePath, const std::string &outputPath) {
  // Load the input image
  // Load the image (in BGR format by default)
  cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
  if (inputImage.empty()) {
    std::cerr << "Error: Unable to load the image." << std::endl;
    return;
  }

  std::vector<cv::KeyPoint> keypoints;
  std::vector<std::vector<cv::DMatch>> matches;

  cv::Mat outputImage = inputImage;

  gq::getCorrespondencesMixed(inputImage, keypoints, matches);

  std::cout << "Hello ! " << std::endl;

  cv::Mat copyImage = inputImage;
  std::vector<cv::KeyPoint> copyKeypoints = keypoints;

  for (const auto &matcheList : matches) {
    if (matcheList.empty()) {
      std::cerr << "Warning: Empty match list encountered." << std::endl;
      continue;
    }

    // std::cout << matcheList.size() << std::endl;

    cv::drawMatches(inputImage, keypoints, copyImage, copyKeypoints, matcheList,
                    outputImage, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>());

    // for (const auto &match : matcheList) {
    //   // Points from the matches
    //   cv::Point2f pt1 = keypoints[match.queryIdx].pt;
    //   cv::Point2f pt2 = keypoints[match.trainIdx].pt +
    //                     cv::Point2f(static_cast<float>(inputImage.cols), 0);

    //   // Draw a thicker line
    //   cv::line(outputImage, pt1, pt2, cv::Scalar(0, 255, 0),
    //            4); // Adjust thickness as needed
    // }
  }

  // Then, overlay thicker lines on top

  for (unsigned int i = 0; i < keypoints.size(); ++i)
    cv::circle(inputImage, keypoints[i].pt, 10, cv::Scalar(256, 0, 0), 5);

  // Save the result image
  if (!cv::imwrite("../src/PremierTest/output/keypoints.jpg", inputImage)) {
    std::cerr << "Error: Unable to save output image to path: " << outputPath
              << std::endl;
    return;
  }

  // Save the result image
  if (!cv::imwrite(outputPath, outputImage)) {
    std::cerr << "Error: Unable to save output image to path: " << outputPath
              << std::endl;
    return;
  }

  std::cout << "Keypoint correspondences image saved to: " << outputPath
            << std::endl;
}

int main(int argc, char **argv) {

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <input_image_path> <output_image_path>" << std::endl;
    return -1;
  }

  std::string inputImagePath = argv[1];
  std::string outputImagePath = argv[2];

  // Process the input image and save the output
  processImage(inputImagePath, outputImagePath);

  return 0;
}
