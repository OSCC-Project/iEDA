
#pragma once

#include <set>
#include <string>
#include <vector>

#include "MPDB.hh"
#include "Setting.hh"
#include "module/logger/Log.hh"
#include "solver/partition/Hmetis.hh"
#include "solver/partition/Metis.hh"
using std::set;
using std::string;
using std::vector;

namespace ipl::imp {

class MPPartition
{
 public:
  MPPartition(MPDB* mdb, Setting* set) : _mdb(mdb), _set(set){};
  void runPartition();

 private:
  void init();
  vector<int> metisPartition();
  vector<int> hmetisPartition();

  set<FPInst*> findInstAdjacent(FPInst* instance);

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
  int findIndex(FPInst* inst)
  {
    int index = -1;
    auto index_iter = _inst_to_index_map.find(inst);
    if (index_iter != _inst_to_index_map.end()) {
      index = (*index_iter).second;
    }
    return index;
  }
  void buildNewModule(vector<int> partition_result);
  float calculateArea(set<int>, FPInst* macro, int index);

  MPDB* _mdb;
  Setting* _set;
  vector<FPInst*> _unfixed_inst_list;    // instance of unfixed
  map<FPInst*, int> _inst_to_index_map;  // inst -> Metis_vetex_id
  map<int, FPInst*> _index_to_inst_map;  // Metis_vetex_id -> inst
};

}  // namespace ipl::imp