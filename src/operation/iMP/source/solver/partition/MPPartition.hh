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

#include <math.h>
#include <time.h>

#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "MPDB.hh"
#include "Setting.hh"
#include "module/logger/Log.hh"
#include "solver/partition/Hmetis.hh"
#include "solver/partition/Metis.hh"

namespace ipl::imp {

class MPPartition
{
 public:
  MPPartition(MPDB* mdb, Setting* set) : _mdb(mdb), _set(set){};
  void runPartition();

 private:
  void init();
  std::vector<int> metisPartition();
  std::vector<int> hmetisPartition();

  std::set<FPInst*> findInstAdjacent(FPInst* instance);

  // for partition
  void add_inst_index(FPInst* inst, int index) { _inst_to_index_map.emplace(inst, index); }
  void add_index_inst(int index, FPInst* inst) { _index_to_inst_map.emplace(index, inst); }
  FPInst* find_inst(int index);
  int findIndex(FPInst* inst);
  void buildNewModule(std::vector<int> partition_result);
  float calculateArea(std::set<int>, FPInst* macro, int index);

  MPDB* _mdb;
  Setting* _set;
  std::vector<FPInst*> _unfixed_inst_list;    // instance of unfixed
  std::map<FPInst*, int> _inst_to_index_map;  // inst -> Metis_vetex_id
  std::map<int, FPInst*> _index_to_inst_map;  // Metis_vetex_id -> inst
};

}  // namespace ipl::imp