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
 * @file		IdbTerm.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Term information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../../../basic/geometry/IdbGeometry.h"
#include "../../../basic/geometry/IdbLayerShape.h"
#include "../IdbEnum.h"
#include "IdbLayer.h"
#include "IdbOrientTransform.h"

namespace idb {

using std::map;
using std::string;
using std::vector;

class IdbCellMaster;
class IdbPort
{
 public:
  IdbPort();
  ~IdbPort();

  // getter
  IdbPortClass& get_port_class() { return _class; }
  vector<IdbLayerShape*>& get_layer_shape() { return _layer_shape_list; }
  std::vector<IdbVia*>& get_via_list() { return _via_list; }
  // IdbLayer* get_layer() { return _layer; }
  IdbOrient get_orient() { return _orient; }
  IdbCoordinate<int32_t>* get_coordinate() { return _coordinate; }
  IdbCoordinate<int32_t>* get_io_average_coordinate() { return _io_average_coordinate; }
  IdbRect* get_io_bounding_box() { return _io_bounding_box; }
  IdbRect get_bounding_box();

  IdbRect get_transform_bounding_box(IdbCoordinate<int32_t>* coordinate, IdbOrientTransform* db_transform);

  const IdbPlacementStatus& get_placement_status() const { return _placement_status; }
  bool is_placed()
  {
    return (_placement_status > IdbPlacementStatus::kNone && _placement_status < IdbPlacementStatus::kMax ? true : false);
  }
  // setter
  void set_port_class(IdbPortClass port_class) { _class = port_class; }
  void set_port_class(string port_class);
  IdbLayerShape* add_layer_shape(IdbLayerShape* layer_shape = nullptr);
  // void set_layer(IdbLayer* layer) { _layer = layer; }
  void set_orient(IdbOrient orient);
  void set_orient_by_enum(int32_t lef_orient);
  void set_coordinate(int32_t x, int32_t y);
  void set_io_average_coordinate(int32_t x, int32_t y);
  void set_io_bounding_box();

  void set_placement_status(IdbPlacementStatus placement_status) { _placement_status = placement_status; }
  void set_placement_status_fix() { _placement_status = IdbPlacementStatus::kFixed; }
  void set_placement_status_place() { _placement_status = IdbPlacementStatus::kPlaced; }
  void set_placement_status_unplace() { _placement_status = IdbPlacementStatus::kUnplaced; }
  void set_placement_status_cover() { _placement_status = IdbPlacementStatus::kCover; }
  void add_via(IdbVia* via) { _via_list.push_back(via); }

  // operator
  IdbLayer* get_top_layer()
  {
    IdbLayer* layer = nullptr;
    for (IdbLayerShape* layer_shape : _layer_shape_list) {
      if (layer == nullptr) {
        layer = layer_shape->get_layer();
      } else {
        if (layer->get_order() < layer_shape->get_layer()->get_order()) {
          layer = layer_shape->get_layer();
        }
      }
    }

    return layer;
  }

  uint8_t findZOrderTop()
  {
    uint8_t z_order = 0;
    for (IdbLayerShape* layer_shape : _layer_shape_list) {
      IdbLayer* layer = layer_shape->get_layer();
      z_order = std::max(z_order, layer->get_order());
    }

    return z_order;
  }

  uint8_t findZOrderBottom()
  {
    uint8_t z_order = UINT8_MAX;
    for (IdbLayerShape* layer_shape : _layer_shape_list) {
      IdbLayer* layer = layer_shape->get_layer();
      z_order = std::min(z_order, layer->get_order());
    }

    return z_order;
  }

  IdbLayerShape* find_layer_shape(std::string layer)
  {
    for (IdbLayerShape* layer_shape : _layer_shape_list) {
      if (layer_shape->get_layer()->get_name().compare(layer) == 0) {
        return layer_shape;
      }
    }

    return nullptr;
  }

 private:
  vector<IdbLayerShape*> _layer_shape_list;
  vector<IdbVia*> _via_list;
  // IdbLayer* _layer;
  IdbPortClass _class;
  IdbCoordinate<int32_t>* _coordinate;
  IdbCoordinate<int32_t>* _io_average_coordinate;
  IdbRect* _io_bounding_box;
  IdbOrient _orient;
  IdbPlacementStatus _placement_status;
};

/**
 * @description


        Term describe the pin attribute information in lef ref.
 *
 */
class IdbTerm
{
 public:
  IdbTerm();
  ~IdbTerm();

  // getter
  const string& get_name() const { return _name; }
  const IdbConnectDirection& get_direction() const { return _direction; }
  const IdbConnectType& get_type() const { return _type; }
  const IdbTermShape& get_shape() const { return _shape; }
  const IdbPlacementStatus& get_placement_status() const { return _placement_status; }
  bool is_placed()
  {
    return (_placement_status > IdbPlacementStatus::kNone && _placement_status < IdbPlacementStatus::kMax ? true : false);
  }
  vector<IdbPort*>& get_port_list() { return _port_list; }
  int32_t get_port_number() { return _port_list.size(); }
  uint8_t get_top_order();
  IdbLayer* get_top_layer();

  IdbCoordinate<int32_t>& get_average_position() { return _average_position; }
  IdbRect* get_bounding_box() { return _bouding_box; }
  bool is_port_exist() { return _has_port; }
  IdbCellMaster* get_cell_master() { return _cell_master; }
  bool is_special_net() { return _is_special_net; }
  bool is_instance_pin() { return _is_instance; }
  bool is_multi_layer();
  bool is_pdn();
  bool is_power();
  bool is_ground();

  vector<IdbCoordinate<int32_t>*>& get_pa_list() { return _pa_list; }

  // setter
  void set_name(string name) { _name = name; }
  void set_direction(IdbConnectDirection direction) { _direction = direction; }
  void set_direction(string direction);
  void set_type(IdbConnectType type) { _type = type; }
  void set_type(string type);
  void set_shape(IdbTermShape shape) { _shape = shape; }
  void set_shape(string shape);

  void set_placement_status(IdbPlacementStatus placement_status) { _placement_status = placement_status; }
  void set_placement_status_fix() { _placement_status = IdbPlacementStatus::kFixed; }
  void set_placement_status_place() { _placement_status = IdbPlacementStatus::kPlaced; }
  void set_placement_status_unplace() { _placement_status = IdbPlacementStatus::kUnplaced; }
  void set_placement_status_cover() { _placement_status = IdbPlacementStatus::kCover; }
  void set_average_position(int32_t x, int32_t y)
  {
    _average_position.set_x(x);
    _average_position.set_y(y);
  }
  void set_bounding_box(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y) { _bouding_box->set_rect(ll_x, ll_y, ur_x, ur_y); }

  IdbPort* add_port(IdbPort* port = nullptr);

  void set_has_port(bool value) { _has_port = value; }
  void set_cell_master(IdbCellMaster* cell_master) { _cell_master = cell_master; }
  void set_special(bool value) { _is_special_net = value; }
  void set_as_instance_pin() { _is_instance = true; }

  // operator

 private:
  string _name;
  IdbConnectDirection _direction;
  IdbConnectType _type;
  IdbTermShape _shape;
  IdbPlacementStatus _placement_status;
  // mustjoin
  vector<IdbPort*> _port_list;
  vector<IdbCoordinate<int32_t>*> _pa_list;

  IdbCoordinate<int32_t> _average_position;
  IdbRect* _bouding_box;
  IdbCellMaster* _cell_master;
  bool _has_port;
  bool _is_special_net;
  bool _is_instance;
};

}  // namespace idb
