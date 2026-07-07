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

class CornerSpacingRule
{
 public:
  CornerSpacingRule() = default;
  ~CornerSpacingRule() = default;
  int32_t get_width_spacing(int32_t width)
  {
    for (int32_t i = width_spacing_list.size() - 1; i >= 0; i--) {
      if (width > width_spacing_list[i].first) {
        return width_spacing_list[i].second;
      }
    }
    return width_spacing_list[0].second;
  }

  bool has_convex_corner = false;
  bool has_concave_corner = false;
  bool has_except_eol = false;
  int32_t except_eol = -1;
  std::vector<std::pair<int32_t, int32_t>> width_spacing_list;
};

}  // namespace idrc
