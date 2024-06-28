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

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "DataManager.hh"

namespace idb {
class IdbLayer;
class IdbSpecialWireSegment;
class IdbRegularWireSegment;
class IdbBlockageList;
class IdbInstance;
class IdbRect;
class IdbVia;
class IdbLayerCut;
class IdbPin;
class IdbSpecialNet;
class IdbLayerRouting;
class IdbSpecialWire;

enum class SegmentType : int8_t;
enum class IdbWireShapeType : uint8_t;
enum class IdbOrient : uint8_t;

template <typename T>
class IdbCoordinate;
}  // namespace idb

namespace ipnp {

class PNPConfig;

class iPNP
{
 public:
  iPNP();
  iPNP(const std::string &config_file);
  ~iPNP();

  PNPConfig* get_config() { return _pnp_config; }
  
  void initSynthesize();   //Generate initial solution. Based on fastplacer.
  void optimize();  // including calling Evaluator and modify PDN
  void synthesizeNetwork(); //Generate the final PDN for future flow( CTS, routing) to use

  void run();  // According to the config. e.g. which Evaluator, which opt algorithm.

 private:
  PNPConfig* _pnp_config = nullptr;
  idb::IdbSpecialNet* _input_netlist;
  GridManager _initialized_network;
  GridManager _current_opt_network;
};

}  // namespace ipnp
