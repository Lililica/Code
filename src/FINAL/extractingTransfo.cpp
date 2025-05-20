// #include "extractingTransfo.hpp"

// // Function to generate a random rotation matrix using QR decomposition
// Matrix3d randomRotationMatrix() {
//   Matrix3d A = Matrix3d::Random();
//   HouseholderQR<Matrix3d> qr(A);
//   Matrix3d Q = qr.householderQ();
//   if (Q.determinant() < 0) {
//     Q.col(0) *= -1; // Ensure it's a proper rotation matrix
//   }
//   return Q;
// }

// // Function to decompose homography matrix into Rotation, Translation, and
// Scale void decomposeHomography(const Matrix4d &H, Matrix3d &R, Vector3d &t,
//                          double &scale) {
//   Matrix3d M = H.block<3, 3>(0, 0); // Upper-left 3x3 part
//   t = H.block<3, 1>(0, 3);          // Translation vector

//   // Perform SVD to extract scale and rotation
//   JacobiSVD<Matrix3d> svd(M, ComputeFullU | ComputeFullV);
//   R = svd.matrixU() * svd.matrixV().transpose();
//   scale = svd.singularValues().mean();
// }

// // Function to extract axis and angle from a rotation matrix
// void rotationMatrixToAxisAngle(const Matrix3d &R, Vector3d &axis,
//                                double &angle) {
//   angle = acos((R.trace() - 1) / 2.0);

//   if (fabs(angle) < 1e-6) {
//     axis = Vector3d(0, 0, 0);
//     angle = 0;
//     return;
//   }

//   if (fabs(angle - M_PI) < 1e-6) {
//     // Special case for 180-degree rotation
//     axis = Vector3d(sqrt((R(0, 0) + 1) / 2.0), sqrt((R(1, 1) + 1) / 2.0),
//                     sqrt((R(2, 2) + 1) / 2.0));
//     axis[0] *= copysign(1.0, R(2, 1) - R(1, 2));
//     axis[1] *= copysign(1.0, R(0, 2) - R(2, 0));
//     axis[2] *= copysign(1.0, R(1, 0) - R(0, 1));
//   } else {
//     // Normal case
//     axis = Vector3d((R(2, 1) - R(1, 2)) / (2 * sin(angle)),
//                     (R(0, 2) - R(2, 0)) / (2 * sin(angle)),
//                     (R(1, 0) - R(0, 1)) / (2 * sin(angle)));
//   }

//   angle = angle * (180.0 / M_PI); // Convert to degrees
// }

// void HomographyToTransform3D(Matrix4d &H) {

//   // Decompose the homography matrix
//   Matrix3d R;
//   Vector3d t;
//   double scale;
//   decomposeHomography(H, R, t, scale);

//   // Extract axis and angle
//   Vector3d axis;
//   double angle;
//   rotationMatrixToAxisAngle(R, axis, angle);

//   // Print results
//   cout << "Generated Homography Matrix (H):\n" << H << "\n\n";
//   cout << "Extracted Translation: " << t.transpose() << "\n";
//   cout << "Extracted Scale: " << scale << "\n";
//   cout << "Extracted Rotation Matrix:\n" << R << "\n";
//   cout << "Rotation Axis: " << axis.transpose() << "\n";
//   cout << "Rotation Angle: " << angle << " degrees\n";
// }

// // Function to decompose a 2D homography matrix
// void decomposeHomography2D(const Matrix3d &H, Matrix2d &R, Vector2d &t,
//                            double &scale, double &angle) {
//   // Extract translation
//   t = H.block<2, 1>(0, 2);

//   // Extract upper-left 2x2 matrix
//   Matrix2d M = H.block<2, 2>(0, 0);

//   // Use SVD to separate rotation and scale
//   JacobiSVD<Matrix2d> svd(M, ComputeFullU | ComputeFullV);
//   R = svd.matrixU() * svd.matrixV().transpose();
//   scale =
//       svd.singularValues().mean(); // Scale factor (average of singular
//       values)

//   // Compute rotation angle
//   angle = atan2(R(1, 0), R(0, 0)) * (180.0 / M_PI); // Convert to degrees
// }

// int main() {
//   // Generate random 2D rotation matrix
//   Matrix2d R_true = randomRotationMatrix();
//   double scale_true =
//       ((double)rand() / RAND_MAX) * 1.5 + 0.5; // Random scale [0.5, 2.0]
//   Vector2d t_true = Vector2d::Random() * 5.0;  // Random translation [-5, 5]

//   // Construct homography matrix H
//   Matrix3d H = Matrix3d::Identity();
//   H.block<2, 2>(0, 0) = scale_true * R_true;
//   H.block<2, 1>(0, 2) = t_true;

//   // Decompose the homography matrix
//   Matrix2d R;
//   Vector2d t;
//   double scale, angle;
//   decomposeHomography2D(H, R, t, scale, angle);

//   // Print results
//   cout << "Generated Homography Matrix (H):\n" << H << "\n\n";
//   cout << "Extracted Translation: " << t.transpose() << "\n";
//   cout << "Extracted Scale: " << scale << "\n";
//   cout << "Extracted Rotation Matrix:\n" << R << "\n";
//   cout << "Rotation Angle: " << angle << " degrees\n";

//   return 0;
// }
