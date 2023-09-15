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
#include <string>
#include <vector>

namespace ieda_contest {

class ContestGuide
{
 public:
  ContestGuide();
  ~ContestGuide() = default;
  // getter
  int get_lb_x() const { return _lb_x; }
  int get_lb_y() const { return _lb_y; }
  int get_rt_x() const { return _rt_x; }
  int get_rt_y() const { return _rt_y; }
  std::string& get_layer_name() { return _layer_name; }
  // setter
  void set_lb_x(const int lb_x) { _lb_x = lb_x; }
  void set_lb_y(const int lb_y) { _lb_y = lb_y; }
  void set_rt_x(const int rt_x) { _rt_x = rt_x; }
  void set_rt_y(const int rt_y) { _rt_y = rt_y; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }

 private:
  int _lb_x;
  int _lb_y;
  int _rt_x;
  int _rt_y;
  std::string _layer_name;
};

class ContestGuideNet
{
 public:
  ContestGuideNet() = default;
  ~ContestGuideNet() = default;
  // getter
  std::string& get_net_name() { return _net_name; }
  std::vector<ContestGuide>& get_guide_list() { return _guide_list; }
  // setter
  void set_net_name(const std::string& net_name) { _net_name = net_name; }
  void set_guide_list(const std::vector<ContestGuide>& guide_list) { _guide_list = guide_list; }
  // function

 private:
  std::string _net_name;
  std::vector<ContestGuide> _guide_list;
};

}  // namespace ieda_contest
