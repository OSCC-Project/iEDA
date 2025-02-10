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

#include <string>
#include <vector>

namespace ipl {
class LGInstance;
class LGInterval;
}  // namespace ipl

namespace ieda_solver {

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
  std::vector<ipl::LGInstance*> get_inst_list() const { return _inst_list; }
  ipl::LGInterval* get_belong_interval() const { return _belong_segment; }
  int32_t get_min_x() const { return _min_x; }
  int32_t get_max_x();
  double get_weight_e() const { return _weight_e; }
  double get_weight_q() const { return _weight_q; }
  int32_t get_total_width() const { return _total_width; }
  std::string get_front_cluster() const { return _front_cluster; }
  std::string get_back_cluster() const { return _back_cluster; }

  // setter
  void set_name(std::string name) { _name = name; }
  void add_inst(ipl::LGInstance* inst) { _inst_list.push_back(inst); }
  void set_belong_interval(ipl::LGInterval* seg) { _belong_segment = seg; }
  void set_min_x(int32_t min_x) { _min_x = min_x; }
  void set_front_cluster(std::string cluster) { _front_cluster = cluster; }
  void set_back_cluster(std::string cluster) {
    _back_cluster = cluster; 
  }


  // function
  void clearAbacusInfo();
  void updateAbacusInfo();
  void insertInstance(ipl::LGInstance* inst);
  void appendCluster(AbacusCluster& cluster);
  void appendInst(ipl::LGInstance* inst);
  void appendInstList(std::vector<ipl::LGInstance*> inst_list);
  int32_t obtainInstIdx(ipl::LGInstance* inst);
  void eraseTargetInstByIdx(int32_t idx);                             
  void eraseTargetInstByIdxPair(int32_t begin_idx, int32_t end_idx);

 private:
  std::string _name;
  std::vector<ipl::LGInstance*> _inst_list;
  ipl::LGInterval* _belong_segment;

  int32_t _min_x;
  double _weight_e;
  double _weight_q;
  int32_t _total_width;

  std::string _front_cluster;
  std::string _back_cluster;
};
}  // namespace ieda_solver
