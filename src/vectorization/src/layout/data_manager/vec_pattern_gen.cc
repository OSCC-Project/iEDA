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
 * @file vec_pattern_gen.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-11-25
 * @brief pattern generation for vectorization
 */

#include "vec_pattern_gen.hh"

#include <ranges>
#include <unordered_set>

#include "log/Log.hh"

namespace ivec {

std::vector<Matrix> VecPatternGenerator::generateAllPatterns(const int& w, const int& h)
{
  // Initialize the state with a 1x1 grid
  PatternState initial;
  initial.grid = Matrix::Zero(1, 1);
  initial.grid(0, 0) = 1;
  initial.current_row = 0;
  initial.current_col = 0;
  initial.max_width = 1;
  initial.max_height = 1;
  initial.val = 1;

  std::vector<Matrix> results;

  // Directions to start, just need to start from top-bottom-right, because of symmetry
  std::vector<PatternDirection> directions = {kTOP, kBOTTOM, kRIGHT};
  std::vector<Matrix> t_b_r;
  generatePatterns(w, h, directions, initial, t_b_r);
  std::ranges::copy(t_b_r, std::back_inserter(results));

  // Generate top-bottom-left patterns by horizontal reversing
  auto t_b_l = std::ranges::transform_view(t_b_r, [](const Matrix& matrix) { return reverse(matrix, true, false); });
  std::ranges::copy(t_b_l, std::back_inserter(results));

  // Generate left-right-top patterns by rotating 90 degrees (counter-clockwise)
  auto l_r_t = std::ranges::transform_view(t_b_r, [](const Matrix& matrix) { return rotate(matrix, 90); });
  std::ranges::copy(l_r_t, std::back_inserter(results));

  // Generate left-right-bottom patterns by rotating 90 degrees (counter-clockwise)
  auto l_r_b = std::ranges::transform_view(t_b_l, [](const Matrix& matrix) { return rotate(matrix, 90); });
  std::ranges::copy(l_r_b, std::back_inserter(results));

  // Drop duplicate patterns/matrices
  dropDuplicatePatterns(results);

  return results;
}

bool VecPatternGenerator::isFeasible(const PatternState& state, const PatternDirection& dir)
{
  // if cuurent row/col is not enough to expand, return true
  switch (dir) {
    case kTOP:
      if (state.current_row == 0) {
        return true;
      }
      return state.grid(state.current_row - 1, state.current_col) == 0;
      break;
    case kBOTTOM:
      if (state.current_row == state.max_height - 1) {
        return true;
      }
      return state.grid(state.current_row + 1, state.current_col) == 0;
      break;
    case kLEFT:
      if (state.current_col == 0) {
        return true;
      }
      return state.grid(state.current_row, state.current_col - 1) == 0;
      break;
    case kRIGHT:
      if (state.current_col == state.max_width - 1) {
        return true;
      }
      return state.grid(state.current_row, state.current_col + 1) == 0;
      break;
  }
  return false;
}

Matrix VecPatternGenerator::reverse(const Matrix& matrix, const bool& horizontal, const bool& vertical)
{
  auto reversed = matrix;
  if (horizontal) {
    reversed = reversed.rowwise().reverse();
  }
  if (vertical) {
    reversed = reversed.colwise().reverse();
  }
  return reversed;
}

Matrix VecPatternGenerator::rotate(const Matrix& matrix, const int& angle)
{
  // angle must be a multiple of 90
  LOG_FATAL_IF(angle % 90 != 0) << "Angle must be a multiple of 90";

  // Calculate the number of 90-degree rotations
  int num_rotations = angle / 90;
  Matrix rotated = matrix;

  // Rotate the matrix
  for (int i = 0; i < num_rotations; ++i) {
    rotated = rotated.transpose().colwise().reverse();
  }

  // rotate 90 degrees counter-clockwise
  return rotated;
}

PatternState VecPatternGenerator::expandGrid(const PatternState& state, const PatternDirection& dir)
{
  auto new_state = state;
  new_state.val += 1;

  // Expand the grid based on the direction
  switch (dir) {
    case kTOP:
      if (new_state.current_row == 0) {
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
      if (new_state.current_row == new_state.max_height - 1) {
        // Add a row at the bottom
        Matrix new_grid(state.grid.rows() + 1, state.grid.cols());
        new_grid.setZero();
        new_grid.block(0, 0, state.grid.rows(), state.grid.cols()) = state.grid;
        new_state.grid = new_grid;
        new_state.max_height += 1;
      }
      break;
    case kLEFT:
      if (new_state.current_col == 0) {
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
      if (new_state.current_col == new_state.max_width - 1) {
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

void VecPatternGenerator::generatePatterns(const int& w, const int& h, const std::vector<PatternDirection>& directions,
                                           const PatternState& current, std::vector<Matrix>& results)
{
  // Add the current grid to results, if it's not 1x1
  if (current.max_width > 1 || current.max_height > 1) {
    results.push_back(current.grid);
  }

  // Try all four possible directions
  for (auto& dir : directions) {
    if (!isFeasible(current, dir)) {
      continue;
    }
    auto new_state = expandGrid(current, dir);

    // Check if expansion is within limits
    if (new_state.max_width > w || new_state.max_height > h) {
      continue;
    }

    // Find the new position to add the pattern
    switch (dir) {
      case kTOP:
        if (new_state.grid(current.current_row - 1, current.current_col) == 0) {
          new_state.grid(current.current_row - 1, current.current_col) = new_state.val;
          new_state.current_row -= 1;
        }
        break;
      case kBOTTOM:
        if (new_state.grid(current.current_row + 1, current.current_col) == 0) {
          new_state.grid(current.current_row + 1, current.current_col) = new_state.val;
          new_state.current_row += 1;
        }
        break;
      case kLEFT:
        if (new_state.grid(current.current_row, current.current_col - 1) == 0) {
          new_state.grid(current.current_row, current.current_col - 1) = new_state.val;
          new_state.current_col -= 1;
        }
        break;
      case kRIGHT:
        if (new_state.grid(current.current_row, current.current_col + 1) == 0) {
          new_state.grid(current.current_row, current.current_col + 1) = new_state.val;
          new_state.current_col += 1;
        }
        break;
    }

    // Recursive call
    generatePatterns(w, h, directions, new_state, results);
  }
}

void VecPatternGenerator::dropDuplicatePatterns(std::vector<Matrix>& results)
{
  // Lambda function to convert a matrix to a string, split by ',' and ';'
  auto to_string = [](const Matrix& matrix) {
    std::string pattern_str;
    for (int i = 0; i < matrix.rows(); ++i) {
      for (int j = 0; j < matrix.cols(); ++j) {
        pattern_str += std::to_string(matrix(i, j));
        if (j < matrix.cols() - 1) {
          pattern_str += ",";
        }
      }
      if (i < matrix.rows() - 1) {
        pattern_str += ";";
      }
    }
    return pattern_str;
  };

  // Use a set to store unique patterns
  std::unordered_set<std::string> unique_matrices;
  std::vector<size_t> to_remove;

  for (size_t i = 0; i < results.size(); ++i) {
    auto pattern_str = to_string(results[i]);
    if (unique_matrices.contains(pattern_str)) {
      to_remove.push_back(i);
    } else {
      unique_matrices.insert(pattern_str);
    }
  }

  // Remove duplicate patterns
  for (size_t i = to_remove.size(); i > 0; --i) {
    results.erase(results.begin() + to_remove[i - 1]);
  }

  return;
}
}  // namespace ivec