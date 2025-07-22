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

class EndOfLineSpacingRule
{
 public:
  EndOfLineSpacingRule() = default;
  ~EndOfLineSpacingRule() = default;
  int32_t eol_spacing = -1;
  int32_t eol_width = -1;
  int32_t eol_within = -1;

  bool has_ete = false;
  /**/ int32_t ete_spacing = -1;

  bool has_par = false;
  /**/ bool has_subtrace_eol_width = false;
  /**/ int32_t par_spacing = -1;
  /**/ int32_t par_within = -1;
  /**/ bool has_two_edges = false;
  /**/ bool has_min_length = false;
  /**/ /**/ int32_t min_length = -1;
  /**/ bool has_same_metal = false;

  bool has_enclose_cut = false;
  /**/ bool has_below = false;
  /**/ bool has_above = false;
  /**/ int32_t enclosed_dist = -1;
  /**/ int32_t cut_to_metal_spacing = -1;
  /**/ bool has_all_cuts = false;
};

}  // namespace idrc
