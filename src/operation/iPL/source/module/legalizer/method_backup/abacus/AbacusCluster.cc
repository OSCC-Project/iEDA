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
 * @Date: 2023-02-03 19:47:46
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-21 11:21:37
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/AbacusCluster.cc
 * @Description:
 *
 *
 */
#include "AbacusCluster.hh"

#include "module/logger/Log.hh"

namespace ipl {

AbacusCluster::AbacusCluster(std::string name)
    : _name(name),
      _belong_segment(nullptr),
      _min_x(INT32_MAX),
      _weight_e(0.0),
      _weight_q(0.0),
      _total_width(0),
      _front_cluster(nullptr),
      _back_cluster(nullptr)
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

void AbacusCluster::insertInstance(LGInstance* inst)
{
  int32_t inst_min_x = inst->get_coordi().get_x();

  int32_t last_min_x = _min_x + _total_width - (*_inst_list.back()).get_shape().get_width();
  if (inst_min_x >= last_min_x) {
    updateAbacusInfo(inst);
    _inst_list.push_back(inst);
    return;
  }

  clearAbacusInfo();
  int32_t front_min_x = _min_x;
  int32_t front_max_x = front_min_x;
  int32_t index = INT32_MIN;
  bool flag_1 = (inst_min_x < front_min_x);
  if (flag_1) {
    updateAbacusInfo(inst);
    index = 0;
  }
  for (size_t i = 0; i < _inst_list.size(); i++) {
    front_max_x += _inst_list[i]->get_shape().get_width();
    updateAbacusInfo(_inst_list.at(i));
    bool flag_2 = (inst_min_x >= front_min_x && inst_min_x < front_max_x);
    if (flag_2) {
      index = i + 1;
      updateAbacusInfo(inst);
    }
    front_min_x = front_max_x;
  }
  if (index == INT32_MIN) {
    LOG_WARNING << "Instance " << inst->get_name() << " cannot insert in the cluster";
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

void AbacusCluster::updateAbacusInfo(LGInstance* inst)
{
  _weight_e += inst->get_weight();
  _weight_q += inst->get_weight() * (inst->get_coordi().get_x() - _total_width);
  _total_width += inst->get_shape().get_width();
}

}  // namespace ipl