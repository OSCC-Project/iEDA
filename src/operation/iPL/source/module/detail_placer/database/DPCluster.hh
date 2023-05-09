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
/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-03-02 11:26:15
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-09 22:22:23
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/database/DPCluster.hh
 * @Description: Cluster of detail placement
 *
 *
 */
#ifndef IPL_DPCLUSTER_H
#define IPL_DPCLUSTER_H

#include <set>
#include <string>
#include <vector>

#include "DPInstance.hh"
#include "DPPin.hh"

namespace ipl {

class DPInterval;

class DPCluster
{
 public:
  DPCluster() = delete;
  explicit DPCluster(std::string name);

  DPCluster(const DPCluster& other)
  {
    _name = other._name;
    _inst_list = other._inst_list;
    _bound_list = other._bound_list;
    _belong_interval = other._belong_interval;
    _min_x = other._min_x;
    _total_width = other._total_width;
    _boundary_min_x = other._boundary_min_x;
    _boundary_max_x = other._boundary_max_x;
    _front_cluster = other._front_cluster;
    _back_cluster = other._back_cluster;
  }
  DPCluster(DPCluster&& other)
  {
    _name = std::move(other._name);
    _inst_list = std::move(other._inst_list);
    _bound_list = std::move(other._bound_list);
    _belong_interval = std::move(other._belong_interval);
    _min_x = std::move(other._min_x);
    _total_width = std::move(other._total_width);
    _boundary_min_x = std::move(other._boundary_min_x);
    _boundary_max_x = std::move(other._boundary_max_x);
    _front_cluster = std::move(other._front_cluster);
    _back_cluster = std::move(other._back_cluster);
  }
  ~DPCluster();

  DPCluster& operator=(const DPCluster& other)
  {
    _name = other._name;
    _inst_list = other._inst_list;
    _bound_list = other._bound_list;
    _belong_interval = other._belong_interval;
    _min_x = other._min_x;
    _total_width = other._total_width;
    _boundary_min_x = other._boundary_min_x;
    _boundary_max_x = other._boundary_max_x;
    _front_cluster = other._front_cluster;
    _back_cluster = other._back_cluster;
    return (*this);
  }
  DPCluster& operator=(DPCluster&& other)
  {
    _name = std::move(other._name);
    _inst_list = std::move(other._inst_list);
    _bound_list = std::move(other._bound_list);
    _belong_interval = std::move(other._belong_interval);
    _min_x = std::move(other._min_x);
    _total_width = std::move(other._total_width);
    _boundary_min_x = std::move(other._boundary_min_x);
    _boundary_max_x = std::move(other._boundary_max_x);
    _front_cluster = std::move(other._front_cluster);
    _back_cluster = std::move(other._back_cluster);
    return (*this);
  }

  // getter
  std::string get_name() const { return _name; }
  const std::vector<DPInstance*>& get_inst_list() const { return _inst_list; }
  DPInterval* get_belong_interval() const { return _belong_interval; }
  int32_t get_min_x() const { return _min_x; }
  int32_t get_max_x() const { return (_min_x + _total_width); }
  int32_t get_total_width() const { return _total_width; }
  int32_t get_boundary_min_x() const { return _boundary_min_x; }
  int32_t get_boundary_max_x() const { return _boundary_max_x; }
  DPCluster* get_front_cluster() const { return _front_cluster; }
  DPCluster* get_back_cluster() const { return _back_cluster; }
  std::vector<int32_t> get_bound_list() const { return _bound_list;}

  // setter
  void set_name(std::string name){ _name = name;}
  void add_inst(DPInstance* inst);
  void set_belong_interval(DPInterval* interval) { _belong_interval = interval; }
  void set_min_x(int32_t min_x);
  void set_boundary_min_x(int32_t b_min_x) { _boundary_min_x = b_min_x; }
  void set_boundary_max_x(int32_t b_max_x) { _boundary_max_x = b_max_x; }
  void set_front_cluster(DPCluster* front_cluster) { _front_cluster = front_cluster; }
  void set_back_cluster(DPCluster* back_cluster) { _back_cluster = back_cluster; }
  void add_bound(int32_t bound);
  void add_bound_list(std::vector<int32_t> bound_list);

  // function
  void insertInstance(DPInstance* inst, int32_t index);
  void replaceInstance(DPInstance* inst, int32_t index);
  void eraseInstance(int32_t index);
  void eraseInstanceRange(int32_t begin_index, int32_t end_index);
  std::pair<int32_t, int32_t> obtainOptimalMinCoordiLine();
  void appendCluster(DPCluster* cluster);

 private:
  std::string _name;
  std::vector<DPInstance*> _inst_list;
  std::vector<int32_t> _bound_list;
  DPInterval* _belong_interval;

  int32_t _min_x;
  int32_t _total_width;

  int32_t _boundary_min_x;
  int32_t _boundary_max_x;
  DPCluster* _front_cluster;
  DPCluster* _back_cluster;
};
}  // namespace ipl
#endif