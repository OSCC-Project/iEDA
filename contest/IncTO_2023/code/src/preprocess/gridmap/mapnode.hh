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
/**
 * @File Name: mapnode.h
 * @Brief : node for gridmap
 * @Author : GuoFan (guofan@ustc.edu)
 * @Version : 1.0
 * @Creat Date : 2023-09-27
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <boost/geometry.hpp>

namespace ieda_contest {

namespace gridmap {

namespace bg = boost::geometry;
namespace bgm = boost::geometry::model;

template <std::size_t DimensionCount>
class Node
{
  typedef bgm::point<int, DimensionCount, bg::cs::cartesian> point;

 public:
  Node() {}
  Node(point position, double supply = 0) : _supply_resource_cnt(supply), _position(position) {}

  void set_supply_resource_cnt(int cnt) { _supply_resource_cnt = cnt; }
  void set_demand_resource_cnt(int cnt) { _demand_resource_cnt = cnt; }
  void increaseDemandResourceCnt() { _demand_resource_cnt++; }
  int remainingResource() { return _supply_resource_cnt - _demand_resource_cnt; }
  const point& get_position() const { return _position; }
  point& set_position() { return _position; }
  void addNetId(int net_id)
  {
    _demanded_net_ids.insert(net_id);
    _demand_resource_cnt = _demanded_net_ids.size();
  }

  bool isWalkable() const { return (_supply_resource_cnt > _demand_resource_cnt); }
  double walkCost()
  {
    return (_supply_resource_cnt > 0 && _supply_resource_cnt > _demand_resource_cnt)
               ? 1.0 + 5.0 * _demand_resource_cnt / _supply_resource_cnt
               : 1000.0 + 500.0 * _demand_resource_cnt / _supply_resource_cnt;
  }

  virtual void refresh() {}

  virtual bool operator==(const Node& n) const
  {
    if (this == &n)
      return true;
    return bg::equals(_position, n.get_position());
  }

  bool operator!=(const Node& n) const { return !(*this == n); }

 private:
  point _position;
  int _demand_resource_cnt = 0;
  int _supply_resource_cnt = 0;
  std::set<long> _demanded_net_ids;
};

}  // namespace gridmap

}  // namespace ieda_contest
