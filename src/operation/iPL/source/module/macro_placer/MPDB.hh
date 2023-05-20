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

#include <fstream>
#include <map>
#include <set>

#include "PlacerDB.hh"
#include "module/logger/Log.hh"
#include "wrapper/DBWrapper.hh"
#include "wrapper/IPLDBWrapper.hh"

namespace ipl::imp {

class MPDB
{
 public:
  MPDB(ipl::PlacerDB* pldb);
  ~MPDB(){};

  // Layout
  const FPLayout* get_layout() const { return _db_wrapper->get_layout(); }

  // Design
  FPDesign* get_design() const { return _db_wrapper->get_design(); }

  // setter
  void add_new_macro(FPInst* new_macro) { _total_macro_list.emplace_back(new_macro); }
  void add_inst_to_new_macro(FPInst* inst, FPInst* new_macro) { _inst_to_new_macro_map.emplace(inst, new_macro); }
  void add_blockage(FPRect* rect) { _blockage_list.emplace_back(rect); }
  void add_guidance_to_macro_name(FPRect* guidance, FPInst* macro);
  void add_guidance_to_macro_name(FPRect* guidance, std::string macro_name);

  // getter
  std::vector<FPInst*> get_total_macro_list() { return _total_macro_list; }
  std::vector<FPInst*> get_place_macro_list() { return _place_macro_list; }
  std::vector<FPNet*> get_new_net_list() { return _new_net_list; }
  int get_ture_index() { return _true_index; }
  std::vector<FPRect*> get_blockage_list() { return _blockage_list; }
  std::map<FPRect*, FPInst*> get_guidance_to_macro_map() { return _guidance_to_macro_map; }

  // Function
  void updatePlaceMacroList();
  void writeBackSourceDataBase() { _db_wrapper->writeBackSourceDataBase(); }
  FPInst* findNewMacro(FPInst* inst);
  FPInst* findMacro(std::string name);
  void buildNetList();
  void writeDB();
  void setMacroFixed(std::string name, int32_t x = -1, int32_t y = -1);
  void writeResult(std::string output_path);

 private:
  void initMPDB();
  void showNetMessage();

  // data
  DBWrapper* _db_wrapper;
  std::map<std::string, FPInst*> _name_to_macro_map;

  int _true_index;  // if i<_true_index, _total_macros[i] is true macro.
  std::vector<FPInst*> _total_macro_list;
  std::vector<FPInst*> _place_macro_list;
  std::vector<FPNet*> _new_net_list;
  std::map<FPInst*, FPInst*> _inst_to_new_macro_map;
  std::vector<FPRect*> _blockage_list;
  std::map<FPRect*, FPInst*> _guidance_to_macro_map;
};
}  // namespace ipl::imp