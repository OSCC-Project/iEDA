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

#include <vector>

#include "PAGCell.hpp"
#include "PAModelStat.hpp"
#include "PANet.hpp"

namespace irt {

class PAModel
{
 public:
  PAModel() = default;
  ~PAModel() = default;
  // getter
  std::vector<GridMap<PAGCell>>& get_layer_gcell_map() { return _layer_gcell_map; }
  std::vector<PANet>& get_pa_net_list() { return _pa_net_list; }
  PAModelStat& get_pa_mode_stat() { return _pa_mode_stat; }
  // setter
  void set_layer_gcell_map(const std::vector<GridMap<PAGCell>>& layer_gcell_map) { _layer_gcell_map = layer_gcell_map; }
  void set_pa_net_list(const std::vector<PANet>& pa_net_list) { _pa_net_list = pa_net_list; }
  void set_pa_mode_stat(const PAModelStat& pa_mode_stat) { _pa_mode_stat = pa_mode_stat; }

 private:
  std::vector<GridMap<PAGCell>> _layer_gcell_map;
  std::vector<PANet> _pa_net_list;
  PAModelStat _pa_mode_stat;
};

}  // namespace irt
