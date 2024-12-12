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
 * @file lm_pattern_op.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-11-28
 * @brief Layout model pattern operation
 */
#pragma once

#include "lm_pattern_gen.hh"
namespace ilm {

class LmPatternOperator
{
 public:
  LmPatternOperator() {}
  ~LmPatternOperator() {}

  void generatePatterns(const int& w, const int& h) { _patterns = LmPatternGenerator::generateAllPatterns(w, h); }

  // Matrix Operator
  static bool isSameSize(const Matrix& matrix1, const Matrix& matrix2);
  static Matrix colReverse(const Matrix& matrix);
  static Matrix rowReverse(const Matrix& matrix);
  static Matrix resize(const Matrix& matrix, const int& target_width, const int& target_height);
  static double similarity(const Matrix& matrix1, const Matrix& matrix2);

 private:
  // Pattern Operator
  void dropHomogenousPatterns();

  std::vector<Matrix> _patterns;
};
}  // namespace ilm