/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-02-03 19:47:46
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-21 11:21:37
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGCluster.cc
 * @Description:
 *
 * Copyright (c) 2023 by iEDA, All Rights Reserved.
 */
#include "LGCluster.hh"

#include "module/logger/Log.hh"

namespace ipl {

LGCluster::LGCluster(std::string name)
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

LGCluster::~LGCluster()
{
}

int32_t LGCluster::get_max_x()
{
  int32_t max_x = _min_x + _total_width;
  return max_x;
}

void LGCluster::clearAbacusInfo()
{
  _weight_e = 0.0;
  _weight_q = 0.0;
  _total_width = 0;
}

void LGCluster::insertInstance(LGInstance* inst)
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

void LGCluster::appendCluster(LGCluster& cluster)
{
  auto other_inst_list = cluster.get_inst_list();
  _inst_list.insert(_inst_list.end(), other_inst_list.begin(), other_inst_list.end());
  _weight_e += cluster.get_weight_e();
  _weight_q += (cluster.get_weight_q() - cluster.get_weight_e() * _total_width);
  _total_width += cluster.get_total_width();
}

void LGCluster::updateAbacusInfo(LGInstance* inst)
{
  _weight_e += inst->get_weight();
  _weight_q += inst->get_weight() * (inst->get_coordi().get_x() - _total_width);
  _total_width += inst->get_shape().get_width();
}

}  // namespace ipl