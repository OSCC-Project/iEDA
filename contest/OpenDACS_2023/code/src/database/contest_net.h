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

#include "contest_guide.h"
#include "contest_pin.h"
#include "contest_segment.h"

namespace ieda_contest {

class ContestNet
{
 public:
  ContestNet() = default;
  ~ContestNet() = default;
  // getter
  std::string get_net_name() { return _net_name; }
  std::vector<ContestGuide>& get_guide_list() { return _guide_list; }
  std::vector<ContestPin>& get_pin_list() { return _pin_list; }
  std::vector<ContestSegment>& get_routing_segment_list() { return _routing_segment_list; }
  // setter
  void set_net_name(const std::string net_name) { _net_name = net_name; }
  void set_guide_list(const std::vector<ContestGuide>& guide_list) { _guide_list = guide_list; }
  void set_pin_list(const std::vector<ContestPin>& pin_list) { _pin_list = pin_list; }
  void set_routing_segment_list(const std::vector<ContestSegment>& routing_segment_list) { _routing_segment_list = routing_segment_list; }
  // function

 private:
  std::string _net_name;
  std::vector<ContestGuide> _guide_list;
  // 下面的数据后续生成
  std::vector<ContestPin> _pin_list;
  std::vector<ContestSegment> _routing_segment_list;
};

}  // namespace ieda_contest
