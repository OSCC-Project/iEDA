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
 * @Date: 2023-02-01 19:36:53
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-17 10:50:00
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGCluster.hh
 * @Description: Instance Clusters of LG
 *
 *
 */
#ifndef IPL_ABACUS_CLUSTER_H
#define IPL_ABACUS_CLUSTER_H

#include <string>
#include <vector>

#include "database/LGInstance.hh"
#include "database/LGInterval.hh"

namespace ipl {

class AbacusCluster
{
 public:
  AbacusCluster() = default;
  explicit AbacusCluster(std::string name);

  AbacusCluster(const AbacusCluster& other)
  {
    _name = other._name;
    _inst_list = other._inst_list;
    _belong_segment = other._belong_segment;
    _min_x = other._min_x;
    _weight_e = other._weight_e;
    _weight_q = other._weight_q;
    _total_width = other._total_width;
    _front_cluster = other._front_cluster;
    _back_cluster = other._back_cluster;
  }
  AbacusCluster(AbacusCluster&& other)
  {
    _name = std::move(other._name);
    _inst_list = std::move(other._inst_list);
    _belong_segment = std::move(other._belong_segment);
    _min_x = std::move(other._min_x);
    _weight_e = std::move(other._weight_e);
    _weight_q = std::move(other._weight_q);
    _total_width = std::move(other._total_width);
    _front_cluster = std::move(other._front_cluster);
    _back_cluster = std::move(other._back_cluster);
  }
  ~AbacusCluster();

  AbacusCluster& operator=(const AbacusCluster& other)
  {
    _name = other._name;
    _inst_list = other._inst_list;
    _belong_segment = other._belong_segment;
    _min_x = other._min_x;
    _weight_e = other._weight_e;
    _weight_q = other._weight_q;
    _total_width = other._total_width;
    _front_cluster = other._front_cluster;
    _back_cluster = other._back_cluster;
    return (*this);
  }
  AbacusCluster& operator=(AbacusCluster&& other)
  {
    _name = std::move(other._name);
    _inst_list = std::move(other._inst_list);
    _belong_segment = std::move(other._belong_segment);
    _min_x = std::move(other._min_x);
    _weight_e = std::move(other._weight_e);
    _weight_q = std::move(other._weight_q);
    _total_width = std::move(other._total_width);
    _front_cluster = std::move(other._front_cluster);
    _back_cluster = std::move(other._back_cluster);
    return (*this);
  }

  // getter
  std::string get_name() const { return _name; }
  std::vector<LGInstance*> get_inst_list() const { return _inst_list; }
  LGInterval* get_belong_interval() const { return _belong_segment; }
  int32_t get_min_x() const { return _min_x; }
  int32_t get_max_x();
  double get_weight_e() const { return _weight_e; }
  double get_weight_q() const { return _weight_q; }
  int32_t get_total_width() const { return _total_width; }
  AbacusCluster* get_front_cluster() const { return _front_cluster; }
  AbacusCluster* get_back_cluster() const { return _back_cluster; }

  // setter
  void set_name(std::string name) { _name = name; }
  void add_inst(LGInstance* inst) { _inst_list.push_back(inst); }
  void set_belong_interval(LGInterval* seg) { _belong_segment = seg; }
  void set_min_x(int32_t min_x) { _min_x = min_x; }
  void set_front_cluster(AbacusCluster* cluster) { _front_cluster = cluster; }
  void set_back_cluster(AbacusCluster* cluster) { _back_cluster = cluster; }

  // function
  void clearAbacusInfo();
  void insertInstance(LGInstance* inst);
  void appendCluster(AbacusCluster& cluster);
  void updateAbacusInfo(LGInstance* inst);

 private:
  std::string _name;
  std::vector<LGInstance*> _inst_list;
  LGInterval* _belong_segment;

  int32_t _min_x;
  double _weight_e;
  double _weight_q;
  int32_t _total_width;

  AbacusCluster* _front_cluster;
  AbacusCluster* _back_cluster;
};
}  // namespace ipl
#endif