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
 * @file Qor.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-27
 * @brief CTS evaluation statistics data.
 */

#pragma once

#include <cstddef>
#include <map>
#include <string>

namespace icts {

struct QorCellStats
{
  std::size_t count = 0U;
  double total_area_um2 = 0.0;
  double total_cap_pf = 0.0;
};

struct QorLibCellDistribution
{
  std::string cell_type;
  std::size_t count = 0U;
  double total_area_um2 = 0.0;
};

struct Qor
{
  bool valid = false;
  double top_wirelength_um = 0.0;
  double trunk_wirelength_um = 0.0;
  double leaf_wirelength_um = 0.0;
  double total_wirelength_um = 0.0;
  double max_net_wirelength_um = 0.0;
  double hpwl_top_wirelength_um = 0.0;
  double hpwl_trunk_wirelength_um = 0.0;
  double hpwl_leaf_wirelength_um = 0.0;
  double hpwl_total_wirelength_um = 0.0;
  double hpwl_max_net_wirelength_um = 0.0;
  std::map<std::string, QorCellStats> cell_stats;
  std::map<std::string, QorLibCellDistribution> lib_cell_dist;
};

}  // namespace icts
