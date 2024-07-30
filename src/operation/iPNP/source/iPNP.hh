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
 * @file iPNP.hh
 * @author Xinhao li
 * @brief Top level file of iPNP module.
 * @version 0.1
 * @date 2024-07-15
 */

#pragma once

#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "GridManager.hh"
#include "iPNPIdbWrapper.hh"

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
class IdbDesign;

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
  iPNP() = default;
  iPNP(const std::string& config_file);
  ~iPNP() = default;

  PNPConfig* get_config() { return _pnp_config; }
  GridManager get_initialized_network() { return _initialized_network; }
  GridManager get_current_opt_network() { return _current_opt_network; }

  void readDef(std::vector<std::string> lef_files, std::string def_path);
  void setIdb(idb::IdbDesign* input_idb_design) { _idb_wrapper.set_idb_design(input_idb_design); }

  void getIdbDesignInfo();
  void initSynthesize();
  void optimize();  // including calling Evaluator and modify PDN
  void saveToIdb() { _idb_wrapper.saveToIdb(_current_opt_network); }
  void writeIdbToDef(std::string def_path) { _idb_wrapper.writeIdbToDef(def_path); }

  void run();  // According to the config. e.g. which Evaluator, which opt algorithm.

 private:
  PNPConfig* _pnp_config = nullptr;
  GridManager _input_network;
  GridManager _initialized_network;
  GridManager _current_opt_network;

  int32_t _input_die_width;
  int32_t _input_die_height;
  std::vector<std::pair<std::pair<int32_t, int32_t>, std::pair<int32_t, int32_t>>> _input_macro_coordinate;

  iPNPIdbWrapper _idb_wrapper;
};

}  // namespace ipnp
