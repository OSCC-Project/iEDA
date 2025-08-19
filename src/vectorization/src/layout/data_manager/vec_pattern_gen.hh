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
 * @file vec_pattern_gen.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-11-25
 * @brief pattern generation for vectorization
 */

#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace ivec {
using Matrix = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
// Structure to hold the current state
struct PatternState
{
  Matrix grid;
  int current_row;
  int current_col;
  int max_width;
  int max_height;
  int val;
};

// Directions for expansion
enum PatternDirection
{
  kTOP,
  kBOTTOM,
  kLEFT,
  kRIGHT
};

class VecPatternGenerator
{
 public:
  VecPatternGenerator() = delete;
  ~VecPatternGenerator() {}

  // Main function to initiate pattern generation
  static std::vector<Matrix> generateAllPatterns(const int& w, const int& h);

 private:
  // Check Direction Feasibility
  static bool isFeasible(const PatternState& state, const PatternDirection& dir);

  // Reverse Operation
  static Matrix reverse(const Matrix& matrix, const bool& horizontal, const bool& vertical);

  // Rotate Operation
  static Matrix rotate(const Matrix& matrix, const int& angle);

  // Function to deep copy the grid and expand it in the specified direction
  static PatternState expandGrid(const PatternState& state, const PatternDirection& dir);

  // Recursive function to generate all patterns
  static void generatePatterns(const int& w, const int& h, const std::vector<PatternDirection>& directions, const PatternState& current,
                               std::vector<Matrix>& results);

  // Drop duplicate patterns
  static void dropDuplicatePatterns(std::vector<Matrix>& results);
};
}  // namespace ivec