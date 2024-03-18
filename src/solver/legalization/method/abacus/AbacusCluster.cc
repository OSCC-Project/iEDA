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
#include "AbacusCluster.hh"

#include <iostream>

#include "LGInstance.hh"
#include "LGInterval.hh"
#include "module/logger/Log.hh"

namespace ieda_solver {

AbacusCluster::AbacusCluster(std::string name)
    : _name(name),
      _belong_segment(nullptr),
      _min_x(INT32_MAX),
      _weight_e(0.0),
      _weight_q(0.0),
      _total_width(0),
      _front_cluster(""),
      _back_cluster("")
{
}

AbacusCluster::~AbacusCluster()
{
}

int32_t AbacusCluster::get_max_x()
{
  int32_t max_x = _min_x + _total_width;
  return max_x;
}

void AbacusCluster::clearAbacusInfo()
{
  _weight_e = 0.0;
  _weight_q = 0.0;
  _total_width = 0;
}

void AbacusCluster::insertInstance(ipl::LGInstance* inst)
{
  int32_t inst_min_x = inst->get_coordi().get_x();

  int32_t last_min_x = _min_x + _total_width - (*_inst_list.back()).get_shape().get_width();
  if (inst_min_x >= last_min_x) {
    appendInst(inst);
    _inst_list.push_back(inst);
    return;
  }

  clearAbacusInfo();
  int32_t front_min_x = _min_x;
  int32_t front_max_x = front_min_x;
  int32_t index = INT32_MIN;
  bool flag_1 = (inst_min_x < front_min_x);
  if (flag_1) {
    appendInst(inst);
    index = 0;
  }
  for (size_t i = 0; i < _inst_list.size(); i++) {
    front_max_x += _inst_list[i]->get_shape().get_width();
    appendInst(_inst_list.at(i));
    bool flag_2 = (inst_min_x >= front_min_x && inst_min_x < front_max_x);
    if (flag_2) {
      index = i + 1;
      appendInst(inst);
    }
    front_min_x = front_max_x;
  }
  if (index == INT32_MIN) {
    std::cout << "Instance " << inst->get_name() << " cannot insert in the cluster" << std::endl;
    return;
  }
  _inst_list.insert(std::next(_inst_list.begin(), index), inst);
  return;
}

void AbacusCluster::appendCluster(AbacusCluster& cluster)
{
  auto other_inst_list = cluster.get_inst_list();
  _inst_list.insert(_inst_list.end(), other_inst_list.begin(), other_inst_list.end());
  _weight_e += cluster.get_weight_e();
  _weight_q += (cluster.get_weight_q() - cluster.get_weight_e() * _total_width);
  _total_width += cluster.get_total_width();
}

void AbacusCluster::appendInst(ipl::LGInstance* inst)
{
  _weight_e += inst->get_weight();
  _weight_q += inst->get_weight() * (inst->get_coordi().get_x() - _total_width);
  _total_width += inst->get_shape().get_width();
}

void AbacusCluster::appendInstList(std::vector<ipl::LGInstance*> inst_list){
  for(auto* inst : inst_list){
    this->appendInst(inst);
  }
  _inst_list = inst_list;
}

void AbacusCluster::updateAbacusInfo()
{
  clearAbacusInfo();
  for (size_t i = 0; i < _inst_list.size(); i++) {
    appendInst(_inst_list[i]);
  }
}

int32_t AbacusCluster::obtainInstIdx(ipl::LGInstance* inst)
{
  int32_t inst_idx = -1;
  for (size_t i = 0; i < _inst_list.size(); i++) {
    if (inst == _inst_list[i]) {
      inst_idx = i;
      break;
    }
  }
  return inst_idx;
}

void AbacusCluster::eraseTargetInstByIdx(int32_t idx){
  _inst_list.erase(_inst_list.begin() + idx);
}

void AbacusCluster::eraseTargetInstByIdxPair(int32_t begin_idx, int32_t end_idx){
  _inst_list.erase(_inst_list.begin()+begin_idx, _inst_list.begin() + end_idx + 1);
}

}  // namespace ieda_solver