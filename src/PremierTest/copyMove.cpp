// #include "copyMove.hpp"

#include <cstddef>
#include <iostream>
// #include <utility> // pair
#include <opencv2/imgproc.hpp>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace gq {

void getCorrespondencesMixed(const cv::Mat &image,
                             std::vector<cv::KeyPoint> &keypoints,
                             std::vector<std::vector<cv::DMatch>> &matches) {

  // detecting key points using FAST
  cv::FAST(image, keypoints, 50, true, cv::FastFeatureDetector::TYPE_9_16);

  std::cout << "Keypoint Number : " << keypoints.size() << "\n";

  bool drawcircle = false;

  if (drawcircle)
    for (unsigned int i = 0; i < keypoints.size(); ++i)
      cv::circle(image, keypoints[i].pt, 30, cv::Scalar(256, 0, 0), 10);

  // compute a dense key points where there's no good features
  std::vector<cv::Point2f> pointsDense;

  // Detect good features to track
  cv::goodFeaturesToTrack(image, pointsDense, /*maxCorners=*/1000,
                          /*qualityLevel=*/0.01,
                          /*minDistance=*/5, cv::noArray(), /*blockSize=*/3,
                          /*useHarrisDetector=*/false,
                          /*k=*/0.04);

  std::vector<cv::KeyPoint> keypointsDense;
  for (const auto &pt : pointsDense) {
    keypointsDense.emplace_back(pt, /*size=*/1.0f);
  }

  // merge the good key points and the dense key points
  // keypoints.insert(keypoints.end(), keypointsDense.begin(),
  //                  keypointsDense.end());
  keypoints = keypointsDense;

  // compute a descriptor for each keypoint
  auto descriptorExtractor = cv::ORB::create();
  cv::Mat descriptors;
  descriptorExtractor->compute(image, keypoints, descriptors);

  // match the descriptors
  auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
  matcher->knnMatch(descriptors, descriptors, matches, 3);
}

} // namespace gq