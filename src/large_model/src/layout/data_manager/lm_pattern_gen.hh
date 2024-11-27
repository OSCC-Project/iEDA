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
 * @file lm_pattern_gen.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-11-25
 * @brief pattern generation for large model
 */

#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "lm_wire_pattern.hh"

namespace ilm {
// Structure to hold the current state
struct PatternState
{
  Matrix grid;
  int current_row;
  int current_col;
  int max_width;
  int max_height;
};

// Directions for expansion
enum PatternDirection
{
  kTOP,
  kBOTTOM,
  kLEFT,
  kRIGHT
};

class LmPatternGen
{
 public:
  LmPatternGen() = default;
  ~LmPatternGen() = default;

 private:
  // Function to deep copy the grid and expand it in the specified direction
  PatternState expandGrid(const PatternState& state, const PatternDirection& dir) const;

  // Recursive function to generate all paths
  void generatePaths(const int& w, const int& h, const std::vector<PatternDirection>& directions, const PatternState& current,
                     std::vector<Matrix>& results) const;

  // Main function to initiate path generation
  std::vector<Matrix> generateAllPaths(const int& w, const int& h) const;

  // Drop duplicate paths
  void dropDuplicatePaths(std::vector<Matrix>& results) const;
};
}  // namespace ilm