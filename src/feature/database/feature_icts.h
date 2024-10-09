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

#include <cstdint>

#include "feature_ista.h"

namespace ieda_feature {

struct CTSSummary
{
  int32_t buffer_num;
  double buffer_area;
  int32_t clock_path_min_buffer;
  int32_t clock_path_max_buffer;
  int32_t max_level_of_clock_tree;
  int32_t max_clock_wirelength;
  double total_clock_wirelength;

  std::vector<ClockTiming> clocks_timing;
};

}  // namespace ieda_feature