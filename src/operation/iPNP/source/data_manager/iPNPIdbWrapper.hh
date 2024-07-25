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

namespace idb {
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
  iPNPIdbWrapper(IdbDesign* idb_design) : _idb_design(idb_design) {}

  iPNPIdbWrapper() = default;
  ~iPNPIdbWrapper() = default;

  unsigned createNet(GridManager pnp_network, ipnp::PowerType net_type);

  void readFromIdb();
  void writeToIdb(GridManager pnp_network);
  void set_idb_design(IdbDesign* idb_design) { _idb_design = idb_design; }
  auto* get_idb_design() { return _idb_design; }

  int32_t get_input_die_llx() { return _idb_design->get_layout()->get_die()->get_llx(); }
  int32_t get_input_die_lly() { return _input_die_lly; }
  int32_t get_input_die_urx() { return _input_die_urx; }
  int32_t get_input_die_ury() { return _input_die_ury; }
  int32_t get_input_die_width() { return std::abs(_input_die_urx - _input_die_llx); }
  int32_t get_input_die_height() { return std::abs(_input_die_ury - _input_die_lly); }
  uint64_t get_input_die_area() { return (uint64_t) get_input_die_width() * (uint64_t) get_input_die_height(); }

 private:
  int32_t _input_die_llx;
  int32_t _input_die_lly;
  int32_t _input_die_urx;
  int32_t _input_die_ury;

  IdbDesign* _idb_design = nullptr;
};

}  // namespace ipnp
