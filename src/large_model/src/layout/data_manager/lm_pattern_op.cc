// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file lm_pattern_op.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-11-28
 * @brief Layout model pattern operation
 */
#include "lm_pattern_op.hh"

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

namespace ilm {
bool LmPatternOperator::isSameSize(const Matrix& matrix1, const Matrix& matrix2)
{
  return matrix1.rows() == matrix2.rows() && matrix1.cols() == matrix2.cols();
}

Matrix LmPatternOperator::colReverse(const Matrix& matrix)
{
  return matrix.colwise().reverse();
}

Matrix LmPatternOperator::rowReverse(const Matrix& matrix)
{
  return matrix.rowwise().reverse();
}

Matrix LmPatternOperator::resize(const Matrix& matrix, const int& target_width, const int& target_height)
{
  // 1. Convert to cv::Mat
  cv::Mat mat;
  cv::eigen2cv(matrix, mat);
  mat.convertTo(mat, CV_32F);

  // 2. Resize
  cv::Mat resized;
  cv::resize(mat, resized, cv::Size(target_width, target_height));

  // 3. Post-process, round all elements to 0 or 1
  resized.forEach<float>([](float& val, const int* position) { val = val >= 0.5 ? 1.0 : 0.0; });

  // 4. Convert back to Eigen::Matrix
  Matrix resized_matrix;
  cv::cv2eigen(resized, resized_matrix);

  return resized_matrix;
}

double LmPatternOperator::similarity(const Matrix& matrix1, const Matrix& matrix2)
{
  auto left = matrix1;
  auto right = matrix2;

  if (!isSameSize(left, right)) {
    left = resize(left, right.cols(), right.rows());
  }

  // Calculate similarity
  auto diff = (left - right).array().abs();
  auto similarity = 1.0 - diff.sum() / (left.rows() * left.cols());

  return similarity;
}

void LmPatternOperator::dropHomogenousPatterns()
{
  // build similarity matrix, drop homogenous patterns
  std::vector<std::vector<double>> similarity_matrix(_patterns.size(), std::vector<double>(_patterns.size(), 0.0));
  for (size_t i = 0; i < _patterns.size(); ++i) {
    for (size_t j = i + 1; j < _patterns.size(); ++j) {
      similarity_matrix[i][j] = similarity(_patterns[i], _patterns[j]);
    }
  }

  // drop homogenous patterns
  std::vector<bool> to_remove(_patterns.size(), false);
  for (size_t i = 0; i < _patterns.size(); ++i) {
    for (size_t j = i + 1; j < _patterns.size(); ++j) {
      if (similarity_matrix[i][j] == 1.0) {
        to_remove[j] = true;
      }
    }
  }
  std::ranges::remove_if(_patterns, [&to_remove, i = 0](const auto& pattern) mutable { return to_remove[i++]; });
}

}  // namespace ilm