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
  iPNPIdbWrapper() = default;
  ~iPNPIdbWrapper() = default;

  idb::IdbSpecialNet* writeNet(GridManager pnp_network, ipnp::PowerType net_type);

  void readFromIdb(std::string input_def);
  void writeToIdb(GridManager pnp_network);

  GridManager get_input_db_pdn() { return _input_db_pdn; }

 private:
  GridManager _input_db_pdn;
};

}  // namespace ipnp
