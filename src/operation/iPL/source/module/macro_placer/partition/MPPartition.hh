
#pragma once

#include <set>
#include <string>
#include <vector>

#include "MPDB.hh"
#include "Partition.hh"
#include "Setting.hh"

using std::string;
using std::vector;

namespace ipl::imp {

class MPPartition : public Partition
{
 public:
  MPPartition(MPDB* mdb, Setting* set) : _mdb(mdb), _set(set){};
  void runPartition();

 private:
  // for metis partition
  virtual void initMetis() override;
  std::set<FPInst*> findInstAdjacent(FPInst* instance);

  void KMPartition(){};
  // for partition
  void add_inst_index(FPInst* inst, int index) { _inst_to_index_map.emplace(inst, index); }
  void add_index_inst(int index, FPInst* inst) { _index_to_inst_map.emplace(index, inst); }
  FPInst* find_inst(int index)
  {
    FPInst* inst = nullptr;
    auto inst_iter = _index_to_inst_map.find(index);
    if (inst_iter != _index_to_inst_map.end()) {
      inst = (*inst_iter).second;
    }
    return inst;
  }
  int find_index(FPInst* inst)
  {
    int index = -1;
    auto index_iter = _inst_to_index_map.find(inst);
    if (index_iter != _inst_to_index_map.end()) {
      index = (*index_iter).second;
    }
    return index;
  }
  void buildNewModule(MetisParam* metis_param);
  float calculateArea(std::set<int>, FPInst* macro, int index);

  MPDB* _mdb;
  Setting* _set;
  MetisParam* _metis_param;
  map<FPInst*, int> _inst_to_index_map;  // inst -> Metis_vetex_id
  map<int, FPInst*> _index_to_inst_map;  // Metis_vetex_id -> inst
};

}  // namespace ipl::imp