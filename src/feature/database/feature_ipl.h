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

namespace ieda_feature {

struct PLCommonSummary
{
  float place_density;
  int64_t HPWL;
  int64_t STWL;
};

struct LGSummary
{
  PLCommonSummary pl_common_summary;
  int64_t lg_total_movement;
  int64_t lg_max_movement;
};

struct PlaceSummary
{
  int32_t bin_number;
  int32_t bin_size_x;
  int32_t bin_size_y;
  int32_t fix_inst_cnt;
  int32_t instance_cnt;
  int32_t net_cnt;
  int32_t overflow_number;
  float overflow;
  int32_t total_pins;

  PLCommonSummary dplace;
  PLCommonSummary gplace;
  LGSummary lg_summary;
};

}  // namespace ieda_feature