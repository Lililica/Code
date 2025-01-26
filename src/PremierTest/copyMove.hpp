// Header file: copyMove.hpp
#ifndef COPYMOVE_HPP
#define COPYMOVE_HPP

#include <opencv2/opencv.hpp>

namespace gq {

void getCorrespondencesMixed(const cv::Mat &image,
                             std::vector<cv::KeyPoint> &keypoints,
                             std::vector<std::vector<cv::DMatch>> &matches);

} // namespace gq

#endif // COPYMOVE_HPP