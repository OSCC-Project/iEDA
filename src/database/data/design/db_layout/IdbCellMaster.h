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
/**
 * @project		iDB
 * @file		IdbCellMaster.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe macros attribute and hierarchy information in lef ref.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"
#include "../IdbEnum.h"
#include "IdbLayer.h"

namespace idb {

using std::string;
using std::vector;

class IdbTerm;
class IdbObs;
class IdbSite;

class IdbCellMaster
{
 public:
  IdbCellMaster();
  ~IdbCellMaster();

  // getter
  CellMasterType& get_type() { return _type; }
  bool is_ring();
  bool is_cover();
  bool is_block();
  bool is_core();
  bool is_pad();
  bool is_endcap();
  bool is_core_filler();
  bool is_pad_filler();
  bool is_io_cell() { return is_pad() || is_pad_filler(); }
  bool is_logic();
  bool is_spacer();
  string& get_name() { return _name; }
  const bool is_symmetry_x() const { return _symmetry_x; }
  const bool is_symmetry_y() const { return _symmetry_y; }
  bool is_symmetry_R90() { return _symmetry_R90; }
  IdbSite* get_site() { return _site; }
  const int64_t get_origin_x() const { return _origin_x; }
  const int64_t get_origin_y() const { return _origin_y; }
  const uint32_t get_width() const { return _width; }
  const uint32_t get_height() const { return _height; }

  vector<IdbTerm*> get_term_list() { return _term_list; }
  IdbTerm* findTerm(std::string term_name);
  vector<IdbObs*> get_obs_list() { return _obs_list; }

  IdbLayer* get_top_layer();

  // setter
  void set_type(CellMasterType type) { _type = type; }
  void set_type(string type_name);
  void set_type_core_filler();
  void set_type_pad_filler();
  void set_name(string name) { _name = name; }
  void set_symmetry_x(bool value) { _symmetry_x = value; }
  void set_symmetry_y(bool value) { _symmetry_y = value; }
  void set_symmetry_R90(bool value) { _symmetry_R90 = value; }
  void set_site(IdbSite* site) { _site = site; }
  void set_origin_x(int64_t value) { _origin_x = value; }
  void set_origin_y(int64_t value) { _origin_y = value; }
  void set_width(uint32_t value) { _width = value; }
  void set_height(uint32_t value) { _height = value; }

  IdbTerm* add_term(IdbTerm* term = nullptr);
  IdbTerm* add_term(string name);
  int32_t get_term_num() { return _term_list.size(); }

  IdbObs* add_obs(IdbObs* obs = nullptr);

  // operator

 private:
  string _name;
  CellMasterType _type;
  bool _symmetry_x;
  bool _symmetry_y;
  bool _symmetry_R90;
  bool _core_filler;
  bool _pad_filler;
  IdbSite* _site = nullptr;
  /*
  If ORIGIN is given in
  the macro, the macro is shifted by the ORIGIN x, y values first,
  before aligning with the DEF placement point. For example, if
  the ORIGIN is 0, -1, then macro geometry at 0, 1 are shifted
  to 0, 0, and then aligned to the DEF placement point.
  */
  int64_t _origin_x;
  int64_t _origin_y;
  uint32_t _width;
  uint32_t _height;

  vector<IdbTerm*> _term_list;
  vector<IdbObs*> _obs_list;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class IdbCellMasterList
{
 public:
  IdbCellMasterList();
  ~IdbCellMasterList();

  // getter
  vector<IdbCellMaster*>& get_cell_master() { return _master_List; }
  int32_t get_cell_master_num() { return _master_List.size(); }

  // setter
  void reset_cell_master();
  IdbCellMaster* set_cell_master(string name);
  void set_number(uint32_t number) { _number = number; }

  // operator
  IdbCellMaster* find_cell_master(const string& src_name);

  vector<IdbCellMaster*> getCoreFillers(vector<string> name_list);
  vector<IdbCellMaster*> getIOFillers(vector<string> name_list);

 private:
  uint32_t _number;
  std::map<string, IdbCellMaster*> _master_map;
  vector<IdbCellMaster*> _master_List;
};
}  // namespace idb
