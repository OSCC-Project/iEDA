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
 * @file		IdbPins.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Pins information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"
#include "../../../basic/geometry/IdbLayerShape.h"
#include "../IdbEnum.h"
#include "../IdbObject.h"
#include "../db_design/IdbTrackGrid.h"
#include "../db_layout/IdbTerm.h"

namespace idb {

class IdbNet;
class IdbInstance;
class IdbSpecialNet;

class IdbPin : public IdbObject
{
 public:
  IdbPin();
  ~IdbPin();

  // getter
  const std::string get_pin_name() const { return _pin_name; }
  IdbTerm* get_term() { return _io_term; }
  const std::string get_term_name() const { return _io_term->get_name(); }
  const std::string get_net_name() const { return _net_name; }
  bool is_io_pin() { return _is_io_pin; }
  bool is_primary_input();
  bool is_primary_output();
  bool is_flip_flop_clk();
  bool is_Q_output() { return _pin_name.compare("Q") == 0 ? true : false; }
  IdbNet* get_net() { return _net; }
  bool is_net_pin() { return _net == nullptr ? false : true; }
  IdbSpecialNet* get_special_net() { return _special_net; }
  bool is_special_net_pin() { return _special_net == nullptr ? false : true; }
  IdbInstance* get_instance() { return _instance; }
  const IdbOrient get_orient() { return _orient; }
  IdbLayerShape* get_bottom_routing_layer_shape();
  std::vector<IdbVia*>& get_via_list() { return _via_list; }

  IdbCoordinate<int32_t>* get_average_coordinate() { return _average_coordinate; }
  IdbCoordinate<int32_t>* get_location() { return _location; };
  IdbCoordinate<int32_t>* get_grid_coordinate() { return _grid_coordinate; }

  std::vector<IdbLayerShape*>& get_port_box_list() { return _layer_shape_list; }
  bool is_multi_layer();

  // setter
  void set_pin_name(std::string pin_name) { _pin_name = pin_name; }
  IdbTerm* set_term(IdbTerm* term = nullptr);
  void set_as_io() { _is_io_pin = true; }
  void set_net_name(std::string net_name) { _net_name = net_name; }
  void set_net(IdbNet* net) { _net = net; }
  void set_special_net(IdbSpecialNet* net) { _special_net = net; }
  void set_instance(IdbInstance* instance) { _instance = instance; }
  void set_average_coordinate(int32_t x, int32_t y);
  void set_location(int32_t x, int32_t y) { _location->set_xy(x, y); }
  void set_grid_coordinate(int32_t x = -1, int32_t y = -1);
  void set_orient_by_enum(int32_t lef_orient);
  void set_orient(IdbOrient orient = IdbOrient::kN_R0) { _orient = orient; }
  bool set_bounding_box();
  void set_pin_bounding_box();
  void set_port_layer_shape();
  void clear_port_layer_shape();
  void set_port_vias();

  // remove
  void remove_net()
  {
    if (_net != nullptr) {
      _net = nullptr;
      _net_name = "";
    }
  }

  // operator
  bool isConnected() { return is_net_pin(); }
  void adjustIOStripe(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end);
  bool isIntersected(int x, int y, IdbLayer* layer);

 private:
  std::string _pin_name;
  std::string _net_name;
  IdbTerm* _io_term;
  IdbNet* _net;
  IdbSpecialNet* _special_net;
  IdbInstance* _instance;
  IdbCoordinate<int32_t>* _average_coordinate;
  IdbCoordinate<int32_t>* _location;  /// the coordinate placed in def
  IdbCoordinate<int32_t>* _grid_coordinate;
  IdbOrient _orient;
  bool _is_io_pin;
  bool _b_new_term;
  std::vector<IdbLayerShape*> _layer_shape_list;
  std::vector<IdbVia*> _via_list;

  bool calculateGridCoordinate();
};

class IdbPins
{
 public:
  IdbPins();
  ~IdbPins();

  // getter
  std::vector<IdbPin*>& get_pin_list() { return _pin_list; }
  const uint32_t get_pin_num() { return _pin_list.size(); }
  uint32_t get_net_pin_num();
  uint get_connected_pin_num();
  IdbPin* find_pin(IdbPin* pin);
  IdbPin* find_pin(std::string pin_name, std::string instance_name = "");
  IdbPin* find_pin_by_term(std::string term_name);
  std::pair<IdbPin*, IdbRect*> find_pin_by_coordinate(IdbCoordinate<int32_t>* coordinate, IdbLayer* layer = nullptr);
  IdbPin* find_pin_by_coordinate_list(vector<IdbCoordinate<int32_t>*>& coordinate_list, IdbLayer* layer);
  // setter
  IdbPin* add_pin_list(IdbPin* pin = nullptr);
  IdbPin* add_pin_list(string pin_name);
  void reset();
  void init(int32_t size) { _pin_list.reserve(size); }

  // Operate
  void remove_pin(IdbPin* pin_remove);
  int32_t getIOPortWidth();
  void checkPins();

 private:
  std::vector<IdbPin*> _pin_list;
};

}  // namespace idb
