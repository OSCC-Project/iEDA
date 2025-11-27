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

namespace irt {

class TGComParam
{
 public:
  TGComParam() = default;
  TGComParam(int32_t expand_step_num, int32_t expand_step_length, double overflow_unit)
  {
    _expand_step_num = expand_step_num;
    _expand_step_length = expand_step_length;
    _overflow_unit = overflow_unit;
  }
  ~TGComParam() = default;
  // getter
  int32_t get_expand_step_num() const { return _expand_step_num; }
  int32_t get_expand_step_length() const { return _expand_step_length; }
  double get_overflow_unit() const { return _overflow_unit; }
  // setter
  void set_expand_step_num(const int32_t expand_step_num) { _expand_step_num = expand_step_num; }
  void set_expand_step_length(const int32_t expand_step_length) { _expand_step_length = expand_step_length; }
  void set_overflow_unit(const double overflow_unit) { _overflow_unit = overflow_unit; }

 private:
  int32_t _expand_step_num = 0;
  int32_t _expand_step_length = 0;
  double _overflow_unit = 0;
};

}  // namespace irt
