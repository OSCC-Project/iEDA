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
  GridMap<PAGCell>& get_pa_gcell_map() { return _pa_gcell_map; }
  std::vector<PANet>& get_pa_net_list() { return _pa_net_list; }
  PAModelStat& get_pa_model_stat() { return _pa_model_stat; }
  irt_int get_curr_iter() { return _curr_iter; }
  // setter
  void set_pa_gcell_map(const GridMap<PAGCell>& pa_gcell_map) { _pa_gcell_map = pa_gcell_map; }
  void set_pa_net_list(const std::vector<PANet>& pa_net_list) { _pa_net_list = pa_net_list; }
  void set_pa_model_stat(const PAModelStat& pa_model_stat) { _pa_model_stat = pa_model_stat; }
  void set_curr_iter(const irt_int curr_iter) { _curr_iter = curr_iter; }

 private:
  GridMap<PAGCell> _pa_gcell_map;
  std::vector<PANet> _pa_net_list;
  PAModelStat _pa_model_stat;
  irt_int _curr_iter = -1;
};

}  // namespace irt
