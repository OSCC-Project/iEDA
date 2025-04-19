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

#include "GridMap.hpp"
#include "PPBox.hpp"
#include "PPIterParam.hpp"
#include "PPNet.hpp"
#include "PPSolution.hpp"

namespace irt {

class PPModel
{
 public:
  PPModel() = default;
  ~PPModel() = default;
  // getter
  std::vector<PPNet>& get_pp_net_list() { return _pp_net_list; }
  int32_t get_iter() const { return _iter; }
  PPIterParam& get_pp_iter_param() { return _pp_iter_param; }
  GridMap<PPBox>& get_pp_box_map() { return _pp_box_map; }
  std::vector<std::vector<PPBoxId>>& get_pp_box_id_list_list() { return _pp_box_id_list_list; }
  // setter
  void set_pp_net_list(const std::vector<PPNet>& pp_net_list) { _pp_net_list = pp_net_list; }
  void set_iter(const int32_t iter) { _iter = iter; }
  void set_pp_iter_param(const PPIterParam& pp_iter_param) { _pp_iter_param = pp_iter_param; }
  void set_pp_box_map(const GridMap<PPBox>& pp_box_map) { _pp_box_map = pp_box_map; }
  void set_pp_box_id_list_list(const std::vector<std::vector<PPBoxId>>& pp_box_id_list_list) { _pp_box_id_list_list = pp_box_id_list_list; }

 private:
  std::vector<PPNet> _pp_net_list;
  int32_t _iter = -1;
  PPIterParam _pp_iter_param;
  GridMap<PPBox> _pp_box_map;
  std::vector<std::vector<PPBoxId>> _pp_box_id_list_list;
};

}  // namespace irt
