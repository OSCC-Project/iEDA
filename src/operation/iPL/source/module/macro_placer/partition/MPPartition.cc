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

#include <math.h>
#include <time.h>

#include <fstream>
#include <string>
#include <vector>

#include "Setting.hh"

using namespace std;
namespace ipl::imp {

void MPPartition::runPartition()
{
  init();
  vector<int> result;
  if (_set->get_partition_type() == PartitionType::Metis) {
    result = metisPartition();
  } else if (_set->get_partition_type() == PartitionType::Hmetis) {
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

vector<int> MPPartition::metisPartition()
{
  // init
  Metis* metis = new Metis();
  // set metis option
  metis->set_ufactor(_set->get_ufactor());  // importance
  metis->set_ncon(_set->get_ncon());
  metis->set_nparts(_set->get_parts());

  clock_t start = clock();
  vector<vector<int>> adjacent_edge_list;
  for (FPInst* inst : _unfixed_inst_list) {
    vector<int> adjacent_index_list;
    set<FPInst*> adjacent_inst_set = findInstAdjacent(inst);
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

  vector<int> result = metis->get_result();
  delete metis;
  return result;
}

vector<int> MPPartition::hmetisPartition()
{
  Hmetis* hmetis = new Hmetis();
  hmetis->set_ufactor(_set->get_ufactor());
  hmetis->set_nparts(_set->get_parts());

  clock_t start = clock();
  vector<vector<int>> hyper_edge_list;
  for (FPNet* net : _mdb->get_design()->get_net_list()) {
    vector<int> hyper_edge;
    set<FPInst*> inst_set = net->get_inst_set();
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
  vector<int> result = hmetis->get_result();
  delete hmetis;
  return result;
}

set<FPInst*> MPPartition::findInstAdjacent(FPInst* inst)
{
  set<FPInst*> adjacent_instance_list;
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
      if (adjacent_inst && adjacent_inst->get_type() == InstType::STD && adjacent_inst != inst) {
        adjacent_instance_list.insert(adjacent_inst);
      }
    }
  }
  return adjacent_instance_list;
}

void MPPartition::buildNewModule(vector<int> partition_result)
{
  vector<set<int>> temp_macro_list;
  temp_macro_list.resize(_set->get_parts());
  int size = partition_result.size();
  for (int i = 0; i < size; ++i) {
    temp_macro_list[partition_result[i]].insert(i);
  }
  LOG_INFO << "part size: ";
  for (set<int> i : temp_macro_list) {
    LOG_INFO << i.size() << " ";
  }

  // buildNewModule
  LOG_INFO << "area of new macros: ";
  int it = 0;
  float new_macro_density = _set->get_new_macro_density();
  float area, total_stdcell_area;
  FPInst* new_macro;
  uint32_t width;
  for (set<int> i : temp_macro_list) {
    new_macro = new FPInst();
    new_macro->set_name("newmacro" + to_string(it));
    total_stdcell_area = calculateArea(i, new_macro, it);
    area = total_stdcell_area / new_macro_density;
    width = uint32_t(sqrt(area));
    new_macro->set_type(InstType::NEWMACRO);
    new_macro->set_width(width);
    new_macro->set_height(width);
    new_macro->set_orient(Orient::N);
    _mdb->add_new_macro(new_macro);
    ++it;
  }
};

float MPPartition::calculateArea(set<int> temp_macro, FPInst* macro, int index)
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