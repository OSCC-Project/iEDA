
#include "MPPartition.hh"

#include <math.h>
#include <time.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Setting.hh"

using namespace std;
namespace ipl::imp {

void MPPartition::runPartition()
{
  if (_set->get_partition_type() == PartitionType::Metis) {
    initMetis();
    MetisPartition(_metis_param);
    buildNewModule(_metis_param);
  } else if (_set->get_partition_type() == PartitionType::KM) {
    KMPartition();
  }
};

void MPPartition::initMetis()
{
  // init
  _metis_param = new MetisParam();
  vector<FPInst*> node_list;  // instance of unfixed

  for (FPInst* inst : _mdb->get_design()->get_std_cell_list()) {
    if (!inst->isFixed()) {
      node_list.emplace_back(inst);
    }
  }
  cout << "Instance Num: " << _mdb->get_design()->get_std_cell_list().size() << endl;
  cout << "node Num: " << node_list.size() << endl;

  clock_t start = clock();
  // mappings: vetex_id <--> inst
  for (size_t i = 0; i < node_list.size(); i++) {
    add_inst_index(node_list[i], i);
    add_index_inst(i, node_list[i]);
  }

  // create adjncy array
  int net_counter = 0;
  int pin_counter = 0;
  _metis_param->_xadj.resize(node_list.size() + 1);
  _metis_param->_part.resize(node_list.size());
  _metis_param->_xadj[net_counter] = 0;
  ++net_counter;
  for (FPInst* node_ptr : node_list) {
    set<int> node_set;
    set<FPInst*> inst_set = findInstAdjacent(node_ptr);
    for (auto ite : inst_set) {
      if (find_index(ite) == -1) {
        continue;
      }
      node_set.insert(find_index(ite));
    }
    if (node_set.size() == 0) {
      _metis_param->_xadj[net_counter] = pin_counter;
      ++net_counter;
      continue;
    } else {
      for (auto num : node_set) {
        _metis_param->_adjncy.emplace_back(num);
        ++pin_counter;
      }
      _metis_param->_xadj[net_counter] = pin_counter;
      ++net_counter;
    }
  }
  cout << "num of verties: " << net_counter << endl;
  cout << "num of edge: " << pin_counter << endl;
  cout << "create adjncy array time consume: " << double(clock() - start) / CLOCKS_PER_SEC << "s" << endl;

  // set metis option
  _metis_param->_options[METIS_OPTION_UFACTOR] = _set->get_ufactor();  // importance
  *_metis_param->_nvtxs = node_list.size();
  *_metis_param->_ncon = _set->get_ncon();
  *_metis_param->_nparts = _set->get_parts();
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

void MPPartition::buildNewModule(MetisParam* metis_param)
{
  vector<set<int>> temp_macro_list;
  temp_macro_list.resize(_set->get_parts());
  int size = metis_param->_part.size();
  for (int i = 0; i < size; ++i) {
    temp_macro_list[metis_param->_part[i]].insert(i);
  }
  cout << "part size: " << endl;
  for (set<int> i : temp_macro_list) {
    cout << i.size() << " ";
  }
  cout << endl;

  // buildNewModule
  cout << "area of new macros: " << endl;
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