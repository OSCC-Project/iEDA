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
#pragma once

#include "RTHeader.hpp"

namespace irt {

class ScaleGrid
{
 public:
  ScaleGrid() = default;
  ~ScaleGrid() = default;
  // getter
  int32_t get_start_line() const { return _start_line; }
  int32_t get_step_length() const { return _step_length; }
  int32_t get_step_num() const { return _step_num; }
  int32_t get_end_line() const { return _end_line; }
  // setter
  void set_start_line(const int32_t start_line) { _start_line = start_line; }
  void set_step_length(const int32_t step_length) { _step_length = step_length; }
  void set_step_num(const int32_t step_num) { _step_num = step_num; }
  void set_end_line(const int32_t end_line) { _end_line = end_line; }
  // function

 private:
  int32_t _start_line = 0;
  int32_t _step_length = 0;
  int32_t _step_num = 0;
  int32_t _end_line = 0;
};

struct CmpScaleGridASC
{
  bool operator()(ScaleGrid& a, ScaleGrid& b) const { return a.get_start_line() < b.get_start_line(); }
};

}  // namespace irt
