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
 * @File Name: contest_guide.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */
#pragma once
#include <set>
#include <string>
#include <vector>

#include "contest_coord.h"

namespace ieda_contest {

class ContestPin
{
 public:
  ContestPin() = default;
  ~ContestPin() = default;
  // getter
  ContestCoord& get_real_coord() { return _real_coord; }
  ContestCoord& get_grid_coord() { return _grid_coord; }
  std::vector<std::string>& get_contained_instance_list() { return _contained_instance_list; }
  std::vector<std::string>& get_contained_pin_list() { return _contained_pin_list; }
  // setter
  void set_real_coord(const ContestCoord& real_coord) { _real_coord = real_coord; }
  void set_grid_coord(const ContestCoord& grid_coord) { _grid_coord = grid_coord; }
  void set_contained_instance_list(const std::vector<std::string>& contained_instance_list)
  {
    _contained_instance_list = contained_instance_list;
  }
  void set_contained_pin_list(const std::vector<std::string>& contained_pin_list) { _contained_pin_list = contained_pin_list; }

 private:
  ContestCoord _real_coord;
  ContestCoord _grid_coord;
  std::vector<std::string> _contained_instance_list;
  std::vector<std::string> _contained_pin_list;
};

}  // namespace ieda_contest
