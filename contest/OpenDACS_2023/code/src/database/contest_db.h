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
 * @File Name: contest_db.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */
#pragma once

#include <map>
#include <vector>

#include "contest_instance.h"
#include "contest_net.h"

namespace ieda_contest {
class ContestDB
{
 public:
  ContestDB();
  ~ContestDB();

  /// getter
  std::map<std::string, int>& get_layer_name_to_idx_map() { return _layer_name_to_idx_map; }
  int get_single_gcell_x_span() const { return _single_gcell_x_span; }
  int get_single_gcell_y_span() const { return _single_gcell_y_span; }
  int get_single_gcell_area() const { return _single_gcell_area; }
  std::map<int, int>& get_layer_gcell_supply_map() { return _layer_gcell_supply_map; }
  std::vector<ContestInstance>& get_instance_list() { return _instance_list; }
  std::vector<ContestNet>& get_net_list() { return _net_list; }
  /// setter
  void set_single_gcell_x_span(const int single_gcell_x_span) { _single_gcell_x_span = single_gcell_x_span; }
  void set_single_gcell_y_span(const int single_gcell_y_span) { _single_gcell_y_span = single_gcell_y_span; }
  void set_single_gcell_area(const int single_gcell_area) { _single_gcell_area = single_gcell_area; }
  // function
  void clear()
  {
    _instance_list.clear();
    std::vector<ContestInstance>().swap(_instance_list);
    _net_list.clear();
    std::vector<ContestNet>().swap(_net_list);
  }

 private:
  std::map<std::string, int> _layer_name_to_idx_map;
  int _single_gcell_x_span = -1;
  int _single_gcell_y_span = -1;
  int _single_gcell_area = -1;
  std::map<int, int> _layer_gcell_supply_map;
  std::vector<ContestInstance> _instance_list;
  std::vector<ContestNet> _net_list;
};

}  // namespace ieda_contest
