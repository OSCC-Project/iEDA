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

namespace ieda_contest {

class ContestGuide
{
 public:
  ContestGuide() = default;
  ~ContestGuide() = default;
  // getter
  int get_lb_x() const { return _lb_x; }
  int get_lb_y() const { return _lb_y; }
  int get_rt_x() const { return _rt_x; }
  int get_rt_y() const { return _rt_y; }
  std::string get_layer_name() { return _layer_name; }
  // setter
  void set_lb_x(const int lb_x) { _lb_x = lb_x; }
  void set_lb_y(const int lb_y) { _lb_y = lb_y; }
  void set_rt_x(const int rt_x) { _rt_x = rt_x; }
  void set_rt_y(const int rt_y) { _rt_y = rt_y; }
  void set_layer_name(const std::string layer_name) { _layer_name = layer_name; }

 private:
  int _lb_x;
  int _lb_y;
  int _rt_x;
  int _rt_y;
  std::string _layer_name;
};

class ContestCoord
{
 public:
  ContestCoord() = default;
  ~ContestCoord() = default;
  // getter
  int get_x() const { return _x; }
  int get_y() const { return _y; }
  int get_layer_id() const { return _layer_id; }
  // setter
  void set_x(const int x) { _x = x; }
  void set_y(const int y) { _y = y; }
  void set_layer_id(const int layer_id) { _layer_id = layer_id; }

 private:
  int _x;
  int _y;
  int _layer_id;
};

struct CmpContestCoord
{
  bool operator()(const ContestCoord& a, const ContestCoord& b) const
  {
    if (a.get_x() != b.get_x()) {
      return a.get_x() < b.get_x();
    } else {
      return a.get_y() != b.get_y() ? a.get_y() < b.get_y() : a.get_layer_id() < b.get_layer_id();
    }
  }
};

class ContestSegment
{
 public:
  ContestSegment() = default;
  ~ContestSegment() = default;
  // getter
  ContestCoord& get_first() { return _first; }
  ContestCoord& get_second() { return _second; }
  // setter
  void set_first(const ContestCoord& first) { _first = first; }
  void set_second(const ContestCoord& second) { _second = second; }

 private:
  ContestCoord _first;
  ContestCoord _second;
};

class ContestPin
{
 public:
  ContestPin() = default;
  ~ContestPin() = default;
  // getter
  ContestCoord& get_coord() { return _coord; }
  std::vector<std::string>& get_contained_instance_list() { return _contained_instance_list; }
  // setter
  void set_coord(const ContestCoord& coord) { _coord = coord; }
  void set_contained_instance_list(const std::vector<std::string>& contained_instance_list)
  {
    _contained_instance_list = contained_instance_list;
  }

 private:
  ContestCoord _coord;
  std::vector<std::string> _contained_instance_list;
};

class ContestInstance
{
 public:
  ContestInstance() = default;
  ~ContestInstance() = default;
  // getter
  std::string& get_name() { return _name; }
  int get_area() const { return _area; }
  // setter
  void set_name(const std::string& name) { _name = name; }
  void set_area(const int area) { _area = area; }

 private:
  std::string _name;
  int _area;
};

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
  std::vector<ContestPin> _pin_list;
  std::vector<ContestSegment> _routing_segment_list;
};

}  // namespace ieda_contest
