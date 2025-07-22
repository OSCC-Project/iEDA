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

#include "DRCHeader.hpp"

namespace idrc {

class MinimumCutRule
{
 public:
  MinimumCutRule() = default;
  ~MinimumCutRule() = default;
  int32_t num_cuts = -1;
  int32_t width = -1;
  bool has_within_cut_distance = false;
  /**/ int32_t within_cut_distance = -1;
  bool has_from_above = false;
  bool has_from_below = false;
  bool has_length = false;
  /**/ int32_t length = -1;
  /**/ int32_t distance = -1;
};

}  // namespace idrc
