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
#include "DPCluster.hh"

#include <algorithm>

namespace ipl {

DPCluster::DPCluster(std::string name)
    : _name(name),
      _belong_interval(nullptr),
      _min_x(INT32_MIN),
      _total_width(0),
      _boundary_min_x(INT32_MIN),
      _boundary_max_x(INT32_MAX),
      _front_cluster(nullptr),
      _back_cluster(nullptr)
{
}

DPCluster::~DPCluster()
{
}

void DPCluster::set_min_x(int32_t min_x)
{
  if (_boundary_min_x + _total_width > _boundary_max_x) {
    LOG_WARNING << "Cluster width is out of interval capcity";
  }

  if (min_x < _boundary_min_x) {
    min_x = _boundary_min_x;
  }
  if (min_x + _total_width > _boundary_max_x) {
    min_x = _boundary_max_x - _total_width;
  }
  _min_x = min_x;
}

void DPCluster::add_inst(DPInstance* inst)
{
  _inst_list.push_back(inst);
  _total_width += inst->get_shape().get_width();
}

void DPCluster::add_bound(int32_t bound)
{
  // // control _bound_list size
  // if(_bound_list.size() > 16){
  //     return;
  // }

  _bound_list.push_back(bound);
}

void DPCluster::add_bound_list(std::vector<int32_t> bound_list)
{
  // // control _bound_list size
  // int32_t remain_size = 16 - _bound_list.size();
  // if(bound_list.size() > remain_size){
  //     _bound_list.insert(_bound_list.end(), bound_list.begin(),
  //     bound_list.begin() + remain_size);
  // }else{
  //     _bound_list.insert(_bound_list.end(), bound_list.begin(),
  //     bound_list.end());
  // }

  _bound_list.insert(_bound_list.end(), bound_list.begin(), bound_list.end());
}

void DPCluster::insertInstance(DPInstance* inst, int32_t index)
{
  // tmp fix bug
  if (_inst_list.empty() && index == 0) {
    _inst_list.push_back(inst);
    _total_width += inst->get_shape().get_width();
    inst->set_internal_id(0);
    return;
  }

  // range check
  if (index < 0 || index >= static_cast<int32_t>(_inst_list.size())) {
    LOG_WARNING << "insertInstance index is out of range";
    return;
  }

  auto it = _inst_list.begin();
  _inst_list.insert(std::next(it, index), inst);
  _total_width += inst->get_shape().get_width();

  for (size_t i = index; i < _inst_list.size(); i++) {
    _inst_list[i]->set_internal_id(i);
  }
}

void DPCluster::replaceInstance(DPInstance* inst, int32_t index)
{
  // range check
  if (index < 0 || index >= static_cast<int32_t>(_inst_list.size())) {
    LOG_WARNING << "replaceInstance index is out of range";
    return;
  }

  auto* origin_inst = _inst_list[index];
  _total_width -= (origin_inst->get_shape().get_width() - inst->get_shape().get_width());

  _inst_list[index] = inst;
}

void DPCluster::eraseInstance(int32_t index)
{
  // range check
  if (index < 0 || index >= static_cast<int32_t>(_inst_list.size())) {
    LOG_WARNING << "eraseInstance index is out of range";
    return;
  }

  auto it = _inst_list.begin();
  std::advance(it, index);
  _total_width -= (*it)->get_shape().get_width();
  _inst_list.erase(it);

  for (size_t i = index; i < _inst_list.size(); i++) {
    _inst_list[i]->set_internal_id(i);
  }
}

void DPCluster::eraseInstanceRange(int32_t begin_index, int32_t end_index)
{
  // range check
  if (begin_index > end_index) {
    LOG_WARNING << "begin_index is grater than end_index";
    return;
  }

  int32_t inst_size = _inst_list.size();
  if (begin_index < 0 || end_index < 0 || begin_index >= inst_size || end_index >= inst_size) {
    LOG_WARNING << "eraseInstance index is out of range";
    return;
  }

  int32_t erase_width = 0;
  for (int32_t i = begin_index; i < inst_size; i++) {
    erase_width += _inst_list[i]->get_shape().get_width();
  }
  _total_width -= erase_width;
  _inst_list.erase(std::next(_inst_list.begin(), begin_index), std::next(_inst_list.begin(), end_index + 1));

  for (size_t i = begin_index; i < _inst_list.size(); i++) {
    _inst_list[i]->set_internal_id(i);
  }
}

std::pair<int32_t, int32_t> DPCluster::obtainOptimalMinCoordiLine()
{
  int32_t bound_size = _bound_list.size();
  if (bound_size == 0) {
    return std::make_pair(_min_x, _min_x);
  }
  std::sort(_bound_list.begin(), _bound_list.end());

  int32_t center_index = (bound_size - 1) / 2;
  int32_t bound_min_x = _bound_list[center_index];
  if (bound_size % 2 == 0) {
    int32_t bound_max_x = _bound_list[center_index + 1];
    return std::make_pair(bound_min_x, bound_max_x);
  } else {
    return std::make_pair(bound_min_x, bound_min_x);
  }
}

void DPCluster::appendCluster(DPCluster* other_cluster)
{
  auto& other_inst_list = other_cluster->get_inst_list();
  _inst_list.insert(_inst_list.end(), other_inst_list.begin(), other_inst_list.end());

  // bound list must be considering offset
  for (int32_t bound : other_cluster->get_bound_list()) {
    this->add_bound(bound - _total_width);
  }
  _total_width += other_cluster->get_total_width();
  _back_cluster = other_cluster->get_back_cluster();
  if (_back_cluster) {
    _back_cluster->set_front_cluster(this);
  }
}

}  // namespace ipl