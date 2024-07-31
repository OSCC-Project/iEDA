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
 * @file NetworkSynthesis.hh
 * @author Xinhao li
 * @brief Synthesize the whole PDN consisting of 3D Template and write DEF file. It doesn't include the function of deciding which template
 * to place in which location. This is determined by the optimizer, or randomly, etc.
 * @version 0.1
 * @date 2024-07-15
 */

#pragma once

#include <fstream>
#include <iostream>

#include "GridManager.hh"
#include "iPNPCommon.hh"

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

enum class SysnType
{
  kDefault,  // synthesize randomly
  kOptimizer,
  kBest,
  kWorst
};

class NetworkSynthesis
{
 public:
  NetworkSynthesis(SysnType sysn_type, GridManager grid_info);
  ~NetworkSynthesis() = default;

  GridManager get_network() { return _synthesized_network; }

  void synthesizeNetwork();
  void randomSys();

 private:
  GridManager _input_grid_info;
  GridManager _synthesized_network;
  SysnType _nework_sys_type;
};

}  // namespace ipnp