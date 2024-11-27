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
 * @file lm_wire_pattern.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-11-25
 * @brief generate wire pattern for large model
 */

#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace ilm {
using Matrix = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

class LmWirePattern
{
 public:
  LmWirePattern() = default;
  LmWirePattern(const std::string& name, const Matrix& matrix) : _pattern_name(name), _pattern_matrix(matrix) {}
  ~LmWirePattern() = default;

  // getter
  std::string get_pattern_name() const { return _pattern_name; }
  Matrix get_pattern_matrix() const { return _pattern_matrix; }

  // setter
  void set_pattern_name(const std::string& name) { _pattern_name = name; }
  void set_pattern_matrix(const Matrix& matrix) { _pattern_matrix = matrix; }

 private:
  std::string _pattern_name = "";

  Matrix _pattern_matrix;
};
}  // namespace ilm