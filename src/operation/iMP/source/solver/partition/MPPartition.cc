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

#include "MPPartition.hh"

namespace ipl::imp {

void MPPartition::runPartition()
{
  init();
  std::vector<int> result;
  if (_set->get_partition_type() == PartitionType::kMetis) {
    result = metisPartition();
  } else if (_set->get_partition_type() == PartitionType::kHmetis) {
    result = hmetisPartition();
  }
  buildNewModule(result);
};

void MPPartition::init()
{
  for (FPInst* inst : _mdb->get_design()->get_std_cell_list()) {
    if (!inst->isFixed()) {
      _unfixed_inst_list.emplace_back(inst);
    }
  }

  // mappings: vetex_id <--> inst
  for (size_t i = 0; i < _unfixed_inst_list.size(); i++) {
    add_inst_index(_unfixed_inst_list[i], i);
    add_index_inst(i, _unfixed_inst_list[i]);
  }
}

std::vector<int> MPPartition::metisPartition()
{
  // init
  Metis* metis = new Metis();
  // set metis option
  metis->set_ufactor(_set->get_ufactor());  // importance
  metis->set_ncon(_set->get_ncon());
  metis->set_nparts(_set->get_parts());

  clock_t start = clock();
  std::vector<std::vector<int>> adjacent_edge_list;
  for (FPInst* inst : _unfixed_inst_list) {
    std::vector<int> adjacent_index_list;
    std::set<FPInst*> adjacent_inst_set = findInstAdjacent(inst);
    for (auto ite : adjacent_inst_set) {
      int inst_index = findIndex(ite);
      if (-1 == inst_index) {
        continue;
      } else {
        adjacent_index_list.emplace_back(inst_index);
      }
    }
    adjacent_edge_list.emplace_back(adjacent_index_list);
  }
  LOG_INFO << "create adjncy array time consume: " << double(clock() - start) / CLOCKS_PER_SEC << "s";

  // call metis
  metis->partition(adjacent_edge_list);

  std::vector<int> result = metis->get_result();
  delete metis;
  return result;
}

std::vector<int> MPPartition::hmetisPartition()
{
  Hmetis* hmetis = new Hmetis();
  hmetis->set_ufactor(_set->get_ufactor());
  hmetis->set_nparts(_set->get_parts());

  clock_t start = clock();
  std::vector<std::vector<int>> hyper_edge_list;
  for (FPNet* net : _mdb->get_design()->get_net_list()) {
    std::vector<int> hyper_edge;
    std::set<FPInst*> inst_set = net->get_inst_set();
    for (auto inst_ite : inst_set) {
      int inst_index = findIndex(inst_ite);
      if (-1 == inst_index) {
        continue;
      } else {
        hyper_edge.emplace_back(inst_index);
      }
    }
    if (hyper_edge.size() > 1) {
      hyper_edge_list.emplace_back(hyper_edge);
    }
  }

  LOG_INFO << "create hyper edge list time consume: " << double(clock() - start) / CLOCKS_PER_SEC << "s";

  // call hmetis
  hmetis->partition(_unfixed_inst_list.size(), hyper_edge_list);
  std::vector<int> result = hmetis->get_result();
  delete hmetis;
  return result;
}

std::set<FPInst*> MPPartition::findInstAdjacent(FPInst* inst)
{
  std::set<FPInst*> adjacent_instance_list;
  FPInst* adjacent_inst;
  for (FPPin* pin : inst->get_pin_list()) {
    if (pin == nullptr) {
      continue;
    }
    if (pin->get_net()->get_pin_list().size() > 10000) {
      continue;
    }
    for (FPPin* adjacent_pin : pin->get_net()->get_pin_list()) {
      adjacent_inst = adjacent_pin->get_instance();
      if (adjacent_inst && adjacent_inst->get_type() == InstType::kStd_cell && adjacent_inst != inst) {
        adjacent_instance_list.insert(adjacent_inst);
      }
    }
  }
  return adjacent_instance_list;
}

FPInst* MPPartition::find_inst(int index)
{
  {
    FPInst* inst = nullptr;
    auto inst_iter = _index_to_inst_map.find(index);
    if (inst_iter != _index_to_inst_map.end()) {
      inst = (*inst_iter).second;
    }
    return inst;
  }
}

int MPPartition::findIndex(FPInst* inst)
{
  {
    int index = -1;
    auto index_iter = _inst_to_index_map.find(inst);
    if (index_iter != _inst_to_index_map.end()) {
      index = (*index_iter).second;
    }
    return index;
  }
}

void MPPartition::buildNewModule(std::vector<int> partition_result)
{
  std::vector<std::set<int>> temp_macro_list;
  temp_macro_list.resize(_set->get_parts());
  int size = partition_result.size();
  for (int i = 0; i < size; ++i) {
    temp_macro_list[partition_result[i]].insert(i);
  }
  LOG_INFO << "part size: ";
  for (std::set<int> i : temp_macro_list) {
    LOG_INFO << i.size() << " ";
  }

  // buildNewModule
  LOG_INFO << "area of new macros: ";
  int it = 0;
  float new_macro_density = _set->get_new_macro_density();
  float area, total_stdcell_area;
  FPInst* new_macro;
  uint32_t width;
  for (std::set<int> i : temp_macro_list) {
    new_macro = new FPInst();
    new_macro->set_name("newmacro" + std::to_string(it));
    total_stdcell_area = calculateArea(i, new_macro, it);
    area = total_stdcell_area / new_macro_density;
    width = uint32_t(sqrt(area));
    new_macro->set_type(InstType::kNew_macro);
    new_macro->set_width(width);
    new_macro->set_height(width);
    new_macro->set_orient(Orient::kN);
    _mdb->add_new_macro(new_macro);
    ++it;
  }
};

float MPPartition::calculateArea(std::set<int> temp_macro, FPInst* macro, int index)
{
  float area = 0;
  // calculate area
  for (int i : temp_macro) {
    FPInst* instance = find_inst(i);
    uint32_t width = instance->get_width();
    uint32_t height = instance->get_height();
    area += float(width) * float(height);

    // IdbInstance -> new_macro
    _mdb->add_inst_to_new_macro(instance, macro);
  }
  return area;
};

}  // namespace ipl::imp