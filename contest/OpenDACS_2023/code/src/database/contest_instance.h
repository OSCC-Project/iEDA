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

class ContestInstance
{
 public:
  ContestInstance() = default;
  ~ContestInstance() = default;
  // getter
  std::string& get_name() { return _name; }
  ContestCoord& get_coord() { return _coord; }
  int get_area() const { return _area; }
  // setter
  void set_name(const std::string& name) { _name = name; }
  void set_coord(const ContestCoord& coord) { _coord = coord; }
  void set_area(const int area) { _area = area; }

 private:
  std::string _name;
  ContestCoord _coord;
  int _area;
};

}  // namespace ieda_contest
