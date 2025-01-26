#include "copyMove.hpp"

#include <iostream>
#include <utility> // pair
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace gq {

// find some good features on the image, compute their descritors and find the
// good matches between them Key idea is to use a feature detection method for
// good features, and to get many other descriptor in a regular grid in
// homogeneous areas, where there's no good features. Then, we look for the 3
// best match, and discard the match that are too near (i.e. selfmatch)
static void
getCorrespondencesMixed(const cv::Mat &image,
                        std::vector<cv::KeyPoint> &keypoints,
                        std::vector<std::vector<cv::DMatch>> &matches) {

  // detecting key points
  // http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html
  // "FAST", "STAR", "ORB", "BRISK", "MSER", "GFTT", "HARRIS", "Dense",
  // "SimpleBlob" cv::Ptr<cv::FeatureDetector> detectorLocal =
  // cv::FeatureDetector::create("FAST");  // opencv 2
  // detectorLocal->detect(image, keypoints); // opencv 2
  cv::FAST(image, keypoints, 1, true, cv::FastFeatureDetector::TYPE_9_16);

  // make a mask of the good keypoints
  cv::Mat mask(image.cols, image.rows, CV_16S, cv::Scalar(1));
  for (unsigned int i = 0; i < keypoints.size(); ++i)
    cv::circle(mask, keypoints[i].pt, 3, cv::Scalar(0), -1);

  // co;pute a dense key points where there's no good features
  std::vector<cv::KeyPoint> keypointsDense;
  //    cv::Ptr<cv::FeatureDetector> detectorDense =
  //    cv::FeatureDetector::create("Dense"); // opencv 2
  //    detectorDense->detect(image, keypointsDense); // opencv 2
  int denseStep = 6;
  int denseSize = 20;
  for (int i = denseSize; i < image.rows - denseSize; i += denseStep)
    for (int j = denseSize; j < image.cols - denseSize; j += denseStep)
      keypointsDense.push_back(cv::KeyPoint(j, i, denseSize));

  // merge the good key points and the dense key points
  keypoints.insert(keypoints.end(), keypointsDense.begin(),
                   keypointsDense.end());

  // compute a descriptor for each keypoint
  // http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_descriptor_extractors.html
  // "SIFT", "SURF", "BRIEF", "BRISK", "ORB", "FREAK"
  // cv::Ptr<cv::DescriptorExtractor> descriptorExtractor =
  // cv::DescriptorExtractor::create("ORB"); // opencv 2
  cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = cv::ORB::create();
  cv::Mat descriptors;
  descriptorExtractor->compute(image, keypoints, descriptors);

  // match the descriptors
  // http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html
  // L1 and L2 norms are preferable choices for SIFT and SURF descriptors
  // HAMMING should be used with ORB, BRISK and BRIEF
  // HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB
  // constructor description).
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(
      "BruteForce-Hamming"); // "BruteForce" (L2), "BruteForce-L1",
                             // "BruteForce-Hamming", "BruteForce-Hamming(2)",
                             // "FlannBased"
  matcher->knnMatch(descriptors, descriptors, matches, 3);
}

// put all the matches in an accumulation buffer to keep only the ones with the
// same point to point direction/lenght. Here we use a double size accumulation
// buffer to represent vectors 'ab' and 'ba' in the same location in the
// accumulation buffer, by forcing the 'i' component to be positive.
static std::vector<std::pair<cv::Point2f, cv::Point2f>>
accumulationBufferMatches(const std::vector<std::vector<cv::DMatch>> &matches,
                          const std::vector<cv::KeyPoint> &keypoints,
                          const int minDistance, const int imageWidth,
                          const int imageHeight) {

  // for more robustness and fast computation, the accumulation buffer size is
  // divided by 'bufferScaleFactor'
  int bufferScaleFactor = 2;

  // This coefficient, multiplyed by the max value found on the accumulation
  // buffer, set a threshold value up to which a set of line is considered as a
  // copy-move.
  float bufferThresholdMaxFactor = 0.8;

  // build the accumulation buffer
  cv::Mat accumulationBuffer(imageHeight / bufferScaleFactor,
                             2 * imageWidth / bufferScaleFactor, CV_32S,
                             cv::Scalar(0));

  // fill the accumulation buffer
  for (unsigned int i = 0; i < matches.size(); ++i)
    for (unsigned int j = 0; j < matches[i].size(); ++j) {

      // compute the distance between the 2 descriptors
      int distance_i = keypoints[matches[i][j].queryIdx].pt.y -
                       keypoints[matches[i][j].trainIdx].pt.y;
      int distance_j = keypoints[matches[i][j].queryIdx].pt.x -
                       keypoints[matches[i][j].trainIdx].pt.x;

      // use a double size accumulation buffer to represent with the same point
      // vector 'ab' and 'ba' by forcing the 'i' component to be positive
      if (distance_i < 0) {
        distance_i = -distance_i;
        distance_j = -distance_j;
      }

      // discard matches that are too near (i.e. self mathing), else update
      // accumulation buffer
      if (sqrt(distance_i * distance_i + distance_j * distance_j) > minDistance)
        accumulationBuffer.at<int>(distance_i / bufferScaleFactor,
                                   (distance_j + imageWidth) /
                                       bufferScaleFactor)++;
    }

  // save the accumulation buffer
  double maxValue = 0.0;
  cv::minMaxLoc(accumulationBuffer, 0, &maxValue);
  // std::cout << "max val " << maxValue << std::endl;
  // cv::imwrite("/tmp/accumulationBuffer.png",(255/(float)maxValue)*accumulationBuffer);

  // find the best values (brute force)
  std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences;
  // for each pixel of the accumulation buffer
  for (int accumulation_i = 0; accumulation_i < accumulationBuffer.rows;
       ++accumulation_i)
    for (int accumulation_j = 0; accumulation_j < accumulationBuffer.cols;
         ++accumulation_j)
      if (accumulationBuffer.at<int>(accumulation_i, accumulation_j) >
          maxValue * bufferThresholdMaxFactor) {

        // get back the shif
        int distance_i_accumulated = bufferScaleFactor * accumulation_i;
        int distance_j_accumulated =
            bufferScaleFactor * accumulation_j - imageWidth;

        // find the corresponding point with the same shift
        for (unsigned int i = 0; i < matches.size(); ++i)
          for (unsigned int j = 0; j < matches[i].size(); ++j) {
            int distance_i = keypoints[matches[i][j].queryIdx].pt.y -
                             keypoints[matches[i][j].trainIdx].pt.y;
            int distance_j = keypoints[matches[i][j].queryIdx].pt.x -
                             keypoints[matches[i][j].trainIdx].pt.x;

            if (distance_i < 0) {
              distance_i = -distance_i;
              distance_j = -distance_j;
            }

            if (fabs(distance_i - distance_i_accumulated) < 2 &&
                fabs(distance_j - distance_j_accumulated) < 2)
              correspondences.push_back(std::pair<cv::Point2f, cv::Point2f>(
                  keypoints[matches[i][j].queryIdx].pt,
                  keypoints[matches[i][j].trainIdx].pt));
          }
      }

  return correspondences;
}

static std::vector<std::pair<cv::Point2f, cv::Point2f>>
getCopyMoveMatches(const cv::Mat &image, cv::Mat &probabilityMap,
                   const bool verbatim, const int minDistance) {
  // find corresponding points
  if (verbatim)
    std::cout << "find keypoints, compute descriptors and get matches ..."
              << std::endl;
  std::vector<cv::KeyPoint> keypoints;
  std::vector<std::vector<cv::DMatch>> matches;
  getCorrespondencesMixed(image, keypoints, matches);
  if (verbatim)
    std::cout << "   nb matches  : " << matches.size() << std::endl;

  // accumulation buffer
  if (verbatim)
    std::cout << "compute the accumulation buffer ..." << std::endl;
  std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences;
  correspondences = accumulationBufferMatches(matches, keypoints, minDistance,
                                              image.cols, image.rows);

  // compute probability map
  probabilityMap = cv::Mat(image.rows, image.cols, CV_64F, cv::Scalar(0.0));
  for (unsigned int i = 0; i < correspondences.size(); ++i) {
    cv::circle(probabilityMap, correspondences[i].first, 6, cv::Scalar(1.0),
               -1);
    cv::circle(probabilityMap, correspondences[i].second, 6, cv::Scalar(1.0),
               -1);
  }

  return correspondences;
}

static float zoomImage(cv::Mat &image, const int imageMaxSize,
                       const bool verbatim) {
  float zoomFactor = 1.0;

  // resize the big images
  if (std::max(image.cols, image.rows) > imageMaxSize) {
    if (verbatim)
      std::cout << "original image size : " << image.cols << " x " << image.rows
                << std::endl;
    zoomFactor = imageMaxSize / (float)std::max(image.cols, image.rows);
    cv::Size size(zoomFactor * image.cols, zoomFactor * image.rows);
    cv::resize(image, image, size, 0, 0, cv::INTER_LANCZOS4);
    std::cout << "resized  image size : " << image.cols << " x " << image.rows
              << std::endl;
  }

  return zoomFactor;
}

cv::Mat copyMove(const cv::Mat &image, cv::Mat &correspondencesImage,
                 const int imageMaxSize, const bool verbatim,
                 const int minDistance) {
  // make a copy of the image
  image.copyTo(correspondencesImage);

  // zoom the image
  float zoomFactor = zoomImage(correspondencesImage, imageMaxSize, verbatim);

  // compute the copy move probability map
  cv::Mat probabilityMap;
  std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences =
      getCopyMoveMatches(correspondencesImage, probabilityMap, verbatim,
                         minDistance);

  // draw correspondences
  for (unsigned int i = 0; i < correspondences.size(); ++i)
    cv::line(correspondencesImage, correspondences[i].first,
             correspondences[i].second, cv::Scalar(0, 0, 255));
  for (unsigned int i = 0; i < correspondences.size(); ++i) {
    cv::circle(correspondencesImage, correspondences[i].first, 3,
               cv::Scalar(255, 0, 0));
    cv::circle(correspondencesImage, correspondences[i].second, 3,
               cv::Scalar(255, 0, 0));
  }

  // unzoom the probability map
  if (zoomFactor != 1.0)
    cv::resize(probabilityMap, probabilityMap, cv::Size(image.cols, image.rows),
               0, 0, cv::INTER_LINEAR);

  return probabilityMap;
}

cv::Mat copyMove(const cv::Mat &image, const int imageMaxSize,
                 const bool verbatim, const int minDistance) {
  // make a copy of the image
  cv::Mat imageCopy;
  image.copyTo(imageCopy);

  // zoom the image
  float zoomFactor = zoomImage(imageCopy, imageMaxSize, verbatim);

  // compute the copy move probability map
  cv::Mat probabilityMap;
  std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences =
      getCopyMoveMatches(imageCopy, probabilityMap, verbatim, minDistance);

  // unzoom the probability map
  if (zoomFactor != 1.0)
    cv::resize(probabilityMap, probabilityMap, cv::Size(image.cols, image.rows),
               0, 0, cv::INTER_LINEAR);

  return probabilityMap;
}

} // end namespace

// put all the matches in an accumulation buffer to keep only the ones with the
// same point-to-point direction/length. Here we use a double-size accumulation
// buffer to represent vectors 'ab' and 'ba' in the same location in the
// accumulation buffer by forcing the 'i' component to be positive.
static std::vector<std::pair<cv::Point2f, cv::Point2f>>
accumulationBufferMatches(const std::vector<std::vector<cv::DMatch>> &matches,
                          const std::vector<cv::KeyPoint> &keypoints,
                          const int minDistance, const int imageWidth,
                          const int imageHeight) {

  // for more robustness and fast computation, the accumulation buffer size is
  // divided by 'bufferScaleFactor'
  int bufferScaleFactor = 2;

  // This coefficient, multiplied by the max value found on the accumulation
  // buffer, sets a threshold value up to which a set of lines is considered as
  // a copy-move.
  float bufferThresholdMaxFactor = 0.8;

  // build the accumulation buffer
  cv::Mat accumulationBuffer(imageHeight / bufferScaleFactor,
                             2 * imageWidth / bufferScaleFactor, CV_32S,
                             cv::Scalar(0));

  // fill the accumulation buffer
  for (unsigned int i = 0; i < matches.size(); ++i)
    for (unsigned int j = 0; j < matches[i].size(); ++j) {

      // compute the distance between the 2 descriptors
      int distance_i = keypoints[matches[i][j].queryIdx].pt.y -
                       keypoints[matches[i][j].trainIdx].pt.y;
      int distance_j = keypoints[matches[i][j].queryIdx].pt.x -
                       keypoints[matches[i][j].trainIdx].pt.x;

      // use a double-size accumulation buffer to represent with the same point
      // vector 'ab' and 'ba' by forcing the 'i' component to be positive
      if (distance_i < 0) {
        distance_i = -distance_i;
        distance_j = -distance_j;
      }

      // discard matches that are too near (i.e., self-matching), else update
      // accumulation buffer
      if (sqrt(distance_i * distance_i + distance_j * distance_j) > minDistance)
        accumulationBuffer.at<int>(distance_i / bufferScaleFactor,
                                   (distance_j + imageWidth) /
                                       bufferScaleFactor)++;
    }

  // save the accumulation buffer
  double maxValue = 0.0;
  cv::minMaxLoc(accumulationBuffer, 0, &maxValue);

  // find the best values (brute force)
  std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences;
  for (int accumulation_i = 0; accumulation_i < accumulationBuffer.rows;
       ++accumulation_i)
    for (int accumulation_j = 0; accumulation_j < accumulationBuffer.cols;
         ++accumulation_j)
      if (accumulationBuffer.at<int>(accumulation_i, accumulation_j) >
          maxValue * bufferThresholdMaxFactor) {

        int distance_i_accumulated = bufferScaleFactor * accumulation_i;
        int distance_j_accumulated =
            bufferScaleFactor * accumulation_j - imageWidth;

        for (unsigned int i = 0; i < matches.size(); ++i)
          for (unsigned int j = 0; j < matches[i].size(); ++j) {
            int distance_i = keypoints[matches[i][j].queryIdx].pt.y -
                             keypoints[matches[i][j].trainIdx].pt.y;
            int distance_j = keypoints[matches[i][j].queryIdx].pt.x -
                             keypoints[matches[i][j].trainIdx].pt.x;

            if (distance_i < 0) {
              distance_i = -distance_i;
              distance_j = -distance_j;
            }

            if (fabs(distance_i - distance_i_accumulated) < 2 &&
                fabs(distance_j - distance_j_accumulated) < 2)
              correspondences.push_back(std::pair<cv::Point2f, cv::Point2f>(
                  keypoints[matches[i][j].queryIdx].pt,
                  keypoints[matches[i][j].trainIdx].pt));
          }
      }

  return correspondences;
}

static std::vector<std::pair<cv::Point2f, cv::Point2f>>
getCopyMoveMatches(const cv::Mat &image, cv::Mat &probabilityMap,
                   const bool verbatim, const int minDistance) {
  if (verbatim)
    std::cout
        << "Finding keypoints, computing descriptors, and getting matches..."
        << std::endl;
  std::vector<cv::KeyPoint> keypoints;
  std::vector<std::vector<cv::DMatch>> matches;
  getCorrespondencesMixed(image, keypoints, matches);
  if (verbatim)
    std::cout << "   Number of matches: " << matches.size() << std::endl;

  if (verbatim)
    std::cout << "Computing the accumulation buffer..." << std::endl;
  std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences;
  correspondences = accumulationBufferMatches(matches, keypoints, minDistance,
                                              image.cols, image.rows);

  probabilityMap = cv::Mat(image.rows, image.cols, CV_64F, cv::Scalar(0.0));
  for (unsigned int i = 0; i < correspondences.size(); ++i) {
    cv::circle(probabilityMap, correspondences[i].first, 6, cv::Scalar(1.0),
               -1);
    cv::circle(probabilityMap, correspondences[i].second, 6, cv::Scalar(1.0),
               -1);
  }

  return correspondences;
}

static float zoomImage(cv::Mat &image, const int imageMaxSize,
                       const bool verbatim) {
  float zoomFactor = 1.0;

  if (std::max(image.cols, image.rows) > imageMaxSize) {
    if (verbatim)
      std::cout << "Original image size: " << image.cols << " x " << image.rows
                << std::endl;
    zoomFactor = imageMaxSize / (float)std::max(image.cols, image.rows);
    cv::Size size(zoomFactor * image.cols, zoomFactor * image.rows);
    cv::resize(image, image, size, 0, 0, cv::INTER_LANCZOS4);
    std::cout << "Resized image size: " << image.cols << " x " << image.rows
              << std::endl;
  }

  return zoomFactor;
}

cv::Mat copyMove(const cv::Mat &image, cv::Mat &correspondencesImage,
                 const int imageMaxSize, const bool verbatim,
                 const int minDistance) {
  image.copyTo(correspondencesImage);

  float zoomFactor = zoomImage(correspondencesImage, imageMaxSize, verbatim);

  cv::Mat probabilityMap;
  std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences =
      getCopyMoveMatches(correspondencesImage, probabilityMap, verbatim,
                         minDistance);

  for (unsigned int i = 0; i < correspondences.size(); ++i)
    cv::line(correspondencesImage, correspondences[i].first,
             correspondences[i].second, cv::Scalar(0, 0, 255));
  for (unsigned int i = 0; i < correspondences.size(); ++i) {
    cv::circle(correspondencesImage, correspondences[i].first, 3,
               cv::Scalar(255, 0, 0));
    cv::circle(correspondencesImage, correspondences[i].second, 3,
               cv::Scalar(255, 0, 0));
  }

  if (zoomFactor != 1.0)
    cv::resize(probabilityMap, probabilityMap, cv::Size(image.cols, image.rows),
               0, 0, cv::INTER_LINEAR);

  return probabilityMap;
}

cv::Mat copyMove(const cv::Mat &image, const int imageMaxSize,
                 const bool verbatim, const int minDistance) {
  cv::Mat imageCopy;
  image.copyTo(imageCopy);

  float zoomFactor = zoomImage(imageCopy, imageMaxSize, verbatim);

  cv::Mat probabilityMap;
  std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences =
      getCopyMoveMatches(imageCopy, probabilityMap, verbatim, minDistance);

  if (zoomFactor != 1.0)
    cv::resize(probabilityMap, probabilityMap, cv::Size(image.cols, image.rows),
               0, 0, cv::INTER_LINEAR);

  return probabilityMap;
}

} // namespace gq
