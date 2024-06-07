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

#include "AccessPoint.hpp"
#include "LayerRect.hpp"
#include "RTHeader.hpp"

namespace irt {

class PAParameter
{
 public:
  PAParameter() = default;
  PAParameter(int32_t enlarged_pitch_num, bool try_adjacent_layer,
              std::function<std::vector<AccessPoint>(int32_t, std::vector<LayerRect>&)> func)

  {
    _enlarged_pitch_num = enlarged_pitch_num;
    _try_adjacent_layer = try_adjacent_layer;
    _func = func;
  }
  ~PAParameter() = default;
  // getter
  int32_t get_enlarged_pitch_num() const { return _enlarged_pitch_num; }
  bool get_try_adjacent_layer() const { return _try_adjacent_layer; }
  std::function<std::vector<AccessPoint>(int32_t, std::vector<LayerRect>&)>& get_func() { return _func; }
  // setter
  void set_enlarged_pitch_num(const int32_t enlarged_pitch_num) { _enlarged_pitch_num = enlarged_pitch_num; }
  void set_try_adjacent_layer(const bool try_adjacent_layer) { _try_adjacent_layer = try_adjacent_layer; }
  void set_func(const std::function<std::vector<AccessPoint>(int32_t, std::vector<LayerRect>&)>& func) { _func = func; }
  // function
 private:
  int32_t _enlarged_pitch_num = 0;
  bool _try_adjacent_layer = 0;
  std::function<std::vector<AccessPoint>(int32_t, std::vector<LayerRect>&)> _func;
};

}  // namespace irt
