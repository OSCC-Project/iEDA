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
 * @file lm_pattern_gen.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-11-25
 * @brief pattern generation for large model
 */

#include "lm_pattern_gen.hh"

#include <unordered_set>

namespace ilm {
PatternState LmPatternGen::expandGrid(const PatternState& state, const PatternDirection& dir) const
{
  auto new_state = state;

  // Expand the grid based on the direction
  switch (dir) {
    case kTOP:
      if (new_state.max_height + 1 > new_state.grid.rows()) {
        // Add a row at the top
        Matrix new_grid(state.grid.rows() + 1, state.grid.cols());
        new_grid.setZero();
        new_grid.block(1, 0, state.grid.rows(), state.grid.cols()) = state.grid;
        new_state.grid = new_grid;
        new_state.current_row += 1;
        new_state.max_height += 1;
      }
      break;
    case kBOTTOM:
      if (new_state.max_height + 1 > new_state.grid.rows()) {
        // Add a row at the bottom
        Matrix new_grid(state.grid.rows() + 1, state.grid.cols());
        new_grid.setZero();
        new_grid.block(0, 0, state.grid.rows(), state.grid.cols()) = state.grid;
        new_state.grid = new_grid;
        new_state.max_height += 1;
      }
      break;
    case kLEFT:
      if (new_state.max_width + 1 > new_state.grid.cols()) {
        // Add a column on the left
        Matrix new_grid(state.grid.rows(), state.grid.cols() + 1);
        new_grid.setZero();
        new_grid.block(0, 1, state.grid.rows(), state.grid.cols()) = state.grid;
        new_state.grid = new_grid;
        new_state.current_col += 1;
        new_state.max_width += 1;
      }
      break;
    case kRIGHT:
      if (new_state.max_width + 1 > new_state.grid.cols()) {
        // Add a column on the right
        Matrix new_grid(state.grid.rows(), state.grid.cols() + 1);
        new_grid.setZero();
        new_grid.block(0, 0, state.grid.rows(), state.grid.cols()) = state.grid;
        new_state.grid = new_grid;
        new_state.max_width += 1;
      }
      break;
  }

  return new_state;
}

void LmPatternGen::generatePaths(const int& w, const int& h, const std::vector<PatternDirection>& directions, const PatternState& current,
                                 std::vector<Matrix>& results) const
{
  // Add the current grid to results, if it's not 1x1
  if (current.max_width > 1 || current.max_height > 1) {
    results.push_back(current.grid);
  }

  // Try all four possible directions
  for (auto& dir : directions) {
    auto new_state = expandGrid(current, dir);

    // Check if expansion is within limits
    if (new_state.max_width > w || new_state.max_height > h)
      continue;

    // Find the new position to add the path
    switch (dir) {
      case kTOP:
        if (new_state.grid(current.current_row - 1, current.current_col) == 0) {
          new_state.grid(current.current_row - 1, current.current_col) = 1;
          new_state.current_row -= 1;
        }
        break;
      case kBOTTOM:
        if (new_state.grid(current.current_row + 1, current.current_col) == 0) {
          new_state.grid(current.current_row + 1, current.current_col) = 1;
          new_state.current_row += 1;
        }
        break;
      case kLEFT:
        if (new_state.grid(current.current_row, current.current_col - 1) == 0) {
          new_state.grid(current.current_row, current.current_col - 1) = 1;
          new_state.current_col -= 1;
        }
        break;
      case kRIGHT:
        if (new_state.grid(current.current_row, current.current_col + 1) == 0) {
          new_state.grid(current.current_row, current.current_col + 1) = 1;
          new_state.current_col += 1;
        }
        break;
    }

    // Recursive call
    generatePaths(w, h, directions, new_state, results);
  }
}

std::vector<Matrix> LmPatternGen::generateAllPaths(const int& w, const int& h) const
{
  std::vector<Matrix> results;

  // Initialize the state with a 1x1 grid
  PatternState initial;
  initial.grid = Matrix::Zero(1, 1);
  initial.grid(0, 0) = 1;
  initial.current_row = 0;
  initial.current_col = 0;
  initial.max_width = 1;
  initial.max_height = 1;

  // Directions to start (just need to start from top-left and top-right corners, because of symmetry)
  std::vector<PatternDirection> left_top = {kLEFT, kTOP};
  std::vector<PatternDirection> right_top = {kRIGHT, kTOP};

  // Start recursion, only generate paths/matrices starting from the top-left and top-right corners
  generatePaths(w, h, left_top, initial, results);
  generatePaths(w, h, right_top, initial, results);

  // Drop duplicate paths/matrices
  dropDuplicatePaths(results);

  return results;
}

void LmPatternGen::dropDuplicatePaths(std::vector<Matrix>& results) const
{
  // Lambda function to convert a matrix to a string, split by ',' and ';'
  auto to_string = [](const Matrix& matrix) {
    std::string path_str;
    for (int i = 0; i < matrix.rows(); ++i) {
      for (int j = 0; j < matrix.cols(); ++j) {
        path_str += std::to_string(matrix(i, j));
        if (j < matrix.cols() - 1) {
          path_str += ",";
        }
      }
      if (i < matrix.rows() - 1) {
        path_str += ";";
      }
    }
    return path_str;
  };

  // Use a set to store unique paths
  std::unordered_set<std::string> unique_matrices;
  std::vector<size_t> to_remove;

  for (size_t i = 0; i < results.size(); ++i) {
    auto path_str = to_string(results[i]);
    if (unique_matrices.contains(path_str)) {
      to_remove.push_back(i);
    } else {
      unique_matrices.insert(path_str);
    }
  }

  // Remove duplicate paths
  for (size_t i = to_remove.size(); i > 0; --i) {
    results.erase(results.begin() + to_remove[i - 1]);
  }

  return;
}
}  // namespace ilm