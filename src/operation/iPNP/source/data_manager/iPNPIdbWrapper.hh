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
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "GridManager.hh"
#include "PowerRouter.hh"
#include "idm.h"

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
  iPNPIdbWrapper() = default;
  ~iPNPIdbWrapper() = default;

  // get die infomation from iDB
  int32_t get_input_die_llx() { return dmInst->get_idb_layout()->get_die()->get_llx(); }  // The smallest x-coordinate of the die rectangle
  int32_t get_input_die_lly() { return dmInst->get_idb_layout()->get_die()->get_lly(); }  // The smallest y-coordinate of the die rectangle
  int32_t get_input_die_urx() { return dmInst->get_idb_layout()->get_die()->get_urx(); }  // The largest x-coordinate of the die rectangle
  int32_t get_input_die_ury() { return dmInst->get_idb_layout()->get_die()->get_ury(); }  // The largest y-coordinate of the die rectangle
  int32_t get_input_die_width() { return dmInst->get_idb_layout()->get_die()->get_width(); }
  int32_t get_input_die_height() { return dmInst->get_idb_layout()->get_die()->get_height(); }
  uint64_t get_input_die_area() { return dmInst->get_idb_layout()->get_die()->get_area(); }

  // get core infomation from iDB
  int32_t get_input_core_lx() { return dmInst->get_idb_layout()->get_core()->get_bounding_box()->get_low_x(); }
  int32_t get_input_core_ly() { return dmInst->get_idb_layout()->get_core()->get_bounding_box()->get_low_y(); }
  int32_t get_input_core_hx() { return dmInst->get_idb_layout()->get_core()->get_bounding_box()->get_high_x(); }
  int32_t get_input_core_hy() { return dmInst->get_idb_layout()->get_core()->get_bounding_box()->get_high_y(); }
  int32_t get_input_core_width() { return dmInst->get_idb_layout()->get_core()->get_bounding_box()->get_width(); }
  int32_t get_input_core_height() { return dmInst->get_idb_layout()->get_core()->get_bounding_box()->get_height(); }
  uint64_t get_input_core_area() { return dmInst->get_idb_layout()->get_core()->get_bounding_box()->get_area(); }

  void saveToIdb(GridManager pnp_network);
  void writeIdbToDef(std::string def_path) { dmInst->saveDef(def_path); }

  void connect_M2_M1_Layer();

 private:
};

}  // namespace ipnp
