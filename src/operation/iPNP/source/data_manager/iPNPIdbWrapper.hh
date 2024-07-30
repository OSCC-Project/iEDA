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
/**
 * @file iPNPIdbWrapper.hh
 * @author Xinhao li
 * @brief
 * @version 0.1
 * @date 2024-07-15
 */

#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "GridManager.hh"
#include "PowerRouter.hh"
#include "iPNPCommon.hh"

namespace idb {
class IdbDesign;
class IdbSpecialNet;
class IdbSpecialNetList;
class IdbSpecialWireList;
class IdbSpecialWire;
class IdbSpecialWireSegment;
class IdbLayer;
class IdbVia;
class IdbPin;
class IdbRect;
class IdbInstance;

enum class SegmentType : int8_t;
enum class IdbWireShapeType : uint8_t;
enum class IdbOrient : uint8_t;

template <typename T>
class IdbCoordinate;
}  // namespace idb

namespace ipnp {

class iPNPIdbWrapper
{
 public:
  iPNPIdbWrapper(idb::IdbDesign* idb_design) : _idb_design(idb_design) {}
  iPNPIdbWrapper() = default;
  ~iPNPIdbWrapper() = default;

  // get die and macro infomation from iDB
  int32_t get_input_die_llx() { return _idb_design->get_layout()->get_die()->get_llx(); }  // The smallest x-coordinate of the die rectangle
  int32_t get_input_die_lly() { return _idb_design->get_layout()->get_die()->get_lly(); }  // The smallest y-coordinate of the die rectangle
  int32_t get_input_die_urx() { return _idb_design->get_layout()->get_die()->get_urx(); }  // The largest x-coordinate of the die rectangle
  int32_t get_input_die_ury() { return _idb_design->get_layout()->get_die()->get_ury(); }  // The largest y-coordinate of the die rectangle
  int32_t get_input_die_width() { return std::abs(get_input_die_urx() - get_input_die_llx()); }
  int32_t get_input_die_height() { return std::abs(get_input_die_ury() - get_input_die_lly()); }
  uint64_t get_input_die_area() { return (uint64_t) get_input_die_width() * (uint64_t) get_input_die_height(); }

  // Question: there is only one IdbCore in IdbLayout?
  int get_input_macro_nums() { return 1; }  // todo
  int32_t get_input_macro_lx() { return _idb_design->get_layout()->get_core()->get_bounding_box()->get_low_x(); }
  int32_t get_input_macro_ly() { return _idb_design->get_layout()->get_core()->get_bounding_box()->get_low_y(); }
  int32_t get_input_macro_hx() { return _idb_design->get_layout()->get_core()->get_bounding_box()->get_high_x(); }
  int32_t get_input_macro_hy() { return _idb_design->get_layout()->get_core()->get_bounding_box()->get_high_y(); }
  int32_t get_input_macro_width() { return std::abs(get_input_macro_hx() - get_input_macro_lx()); }
  int32_t get_input_macro_height() { return std::abs(get_input_macro_hy() - get_input_macro_ly()); }
  uint64_t get_input_macro_area() { return (uint64_t) get_input_macro_width() * (uint64_t) get_input_macro_height(); }

  auto* get_idb_design() { return _idb_design; }
  void set_idb_design(idb::IdbDesign* idb_design) { _idb_design = idb_design; }

  void saveToIdb(GridManager pnp_network);
  void writeIdbToDef(std::string def_file_path);

 private:
  idb::IdbDesign* _idb_design = nullptr;
};

}  // namespace ipnp
