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
 * @file		IdbSpecialNet.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Special Nets information.
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
#include "../db_layout/IdbLayer.h"
#include "IdbInstance.h"
#include "IdbPins.h"
#include "IdbSpecialWire.h"

namespace idb {

using std::string;
using std::vector;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @description
 * Special Net in is used as the power net,
 *
[SPECIALNETS numNets ;
  [– netName
    [ ( {compName pinName | PIN pinName} [+ SYNTHESIZED] ) ] ...
    [+ VOLTAGE volts]
    [specialWiring] ...
    [+ SOURCE {DIST | NETLIST | TIMING | USER}]
    [+ FIXEDBUMP]
    [+ ORIGINAL netName]
    [+ USE {ANALOG | CLOCK | GROUND | POWER | RESET | SCAN | SIGNAL | TIEOFF}]
    [+ PATTERN {BALANCED | STEINER | TRUNK | WIREDLOGIC}]
    [+ ESTCAP wireCapacitance]
    [+ WEIGHT weight]
    [+ PROPERTY {propName propVal} ...] ...
  ;] ...
END SPECIALNETS]

 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class IdbSpecialNet
{
 public:
  IdbSpecialNet();
  ~IdbSpecialNet();

  // getter
  const string& get_net_name() const { return _net_name; }
  const IdbConnectType get_connect_type() const { return _connect_type; }
  const IdbInstanceType get_source_type() const { return _source_type; }
  const string get_original_net_name() const { return _original_net_name; }
  const int32_t get_weight() const { return _weight; }

  int32_t get_segment_num()
  {
    int number = 0;
    for (auto* wire : _wire_list->get_wire_list()) {
      number += wire->get_num();
    }

    return number;
  }

  int32_t get_via_num()
  {
    int number = 0;
    for (auto* wire : _wire_list->get_wire_list()) {
      number += wire->get_via_num();
    }

    return number;
  }

  // bool is_vdd() { return 0 == _net_name.compare("VDD") ? true : false; }
  // bool is_vss() { return 0 == _net_name.compare("VSS") ? true : false; }
  bool is_vdd() { return _connect_type == IdbConnectType::kPower ? true : false; }
  bool is_vss() { return _connect_type == IdbConnectType::kGround ? true : false; }

  IdbPins* get_io_pin_list() { return _io_pin_list; }
  IdbPins* get_instance_pin_list() { return _instance_pin_list; }
  IdbInstanceList* get_instance_list() { return _instance_list; }
  IdbSpecialWireList* get_wire_list() { return _wire_list; }
  vector<string>& get_pin_string_list() { return _pin_string_list; }

  // setter
  void set_net_name(string name) { _net_name = name; }
  void set_connect_type(IdbConnectType type) { _connect_type = type; }
  void set_connect_type(string type);
  void set_original_net_name(string name) { _original_net_name = name; }

  void set_source_type(string type);
  void set_weight(int32_t weight) { _weight = weight; }
  /**
   * @brief Add function set_wire_list()
   */
  void set_wire_list(IdbSpecialWireList* wire_list) { _wire_list = wire_list; }

  void add_io_pin(IdbPin* io_pin);
  auto* get_io_pins() { return _io_pin_list; }
  void add_instance_pin(IdbPin* inst_pin);
  void add_instance(IdbInstance* instance);

  void add_pin_string(string pin_name) { _pin_string_list.emplace_back(pin_name); }

  // operator
  int32_t get_layer_width(string layer_name);
  bool containPin(std::string pin_name)
  {
    for (auto name : _pin_string_list) {
      if (name.compare(pin_name) == 0) {
        return true;
      }
    }
    return false;
  }

 private:
  string _net_name;
  string _original_net_name;

  IdbConnectType _connect_type;
  IdbInstanceType _source_type;
  int32_t _weight;

  IdbPins* _io_pin_list;        // Pins around the chip
  IdbPins* _instance_pin_list;  // Pins around the Macro
  IdbInstanceList* _instance_list;
  IdbSpecialWireList* _wire_list;

  vector<string> _pin_string_list;
};

class IdbSpecialNetList
{
 public:
  IdbSpecialNetList();
  ~IdbSpecialNetList();

  // getter
  vector<IdbSpecialNet*>& get_net_list() { return _net_list; }
  size_t get_num() { return _net_list.size(); }
  IdbSpecialNet* find_net(string name);
  IdbSpecialNet* find_net(size_t index);
  IdbSpecialNetEdgeSegmenArray* find_edge_segment_array_by_layer(IdbLayer* layer);
  size_t get_segment_array_num() { return _edge_segment_list.size(); }
  uint64_t get_segment_num()
  {
    uint64_t number = 0;

    for (auto* net : _net_list) {
      number += net->get_segment_num();
    }

    return number;
  }

  uint get_via_num()
  {
    uint number = 0;

    for (auto* net : _net_list) {
      number += net->get_via_num();
    }

    return number;
  }

  // setter
  IdbSpecialNet* add_net(IdbSpecialNet* net = nullptr);
  IdbSpecialNet* add_net(string name);
  IdbSpecialNetEdgeSegmenArray* add_edge_segment_array_for_layer(IdbLayerRouting* layer);
  IdbSpecialNetEdgeSegmenArray* add_edge_segment_array(IdbSpecialNetEdgeSegmenArray* edge_segment = nullptr);
  void clear_edge_list();

  void resize(size_t size) { _net_list.reserve(size); }

  //   void reset();

  // operator
  IdbSpecialWire* generateWire(string net_name);

  void initEdge(IdbLayers* layers);
  bool connectIO(vector<IdbCoordinate<int32_t>*>& point_list, IdbLayer* layer);
  bool addPowerStripe(vector<IdbCoordinate<int32_t>*>& point_list, string net_name, string layer_name);

 private:
  vector<IdbSpecialNet*> _net_list;
  vector<IdbSpecialNetEdgeSegmenArray*> _edge_segment_list;
};

}  // namespace idb
