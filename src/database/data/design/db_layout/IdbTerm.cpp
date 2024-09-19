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
#include "IdbTerm.h"

#include <limits.h>

#include <algorithm>

namespace idb {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbPort::IdbPort()
{
  _class = IdbPortClass::kNone;
  _coordinate = new IdbCoordinate<int32_t>();
  _io_average_coordinate = new IdbCoordinate<int32_t>();
  _io_bounding_box = new IdbRect();
  _placement_status = IdbPlacementStatus::kNone;
  _orient = IdbOrient::kN_R0;
}

IdbPort::~IdbPort()
{
  for (auto layer_shape : _layer_shape_list) {
    if (layer_shape != nullptr) {
      delete layer_shape;
      layer_shape = nullptr;
    }
  }

  if (_coordinate != nullptr) {
    delete _coordinate;
    _coordinate = nullptr;
  }

  if (_io_average_coordinate != nullptr) {
    delete _io_average_coordinate;
    _io_average_coordinate = nullptr;
  }

  if (_io_bounding_box != nullptr) {
    delete _io_bounding_box;
    _io_bounding_box = nullptr;
  }
}

void IdbPort::set_port_class(string port_class)
{
  _class = IdbEnum::GetInstance()->get_connect_property()->get_port_class(port_class);
}

IdbLayerShape* IdbPort::add_layer_shape(IdbLayerShape* layer_shape)
{
  if (layer_shape == nullptr) {
    layer_shape = new IdbLayerShape();
  }

  _layer_shape_list.emplace_back(layer_shape);

  return layer_shape;
}

IdbRect IdbPort::get_bounding_box()
{
  int32_t bounding_box_ll_x = INT_MAX;
  int32_t bounding_box_ll_y = INT_MAX;
  int32_t bounding_box_ur_x = INT_MIN;
  int32_t bounding_box_ur_y = INT_MIN;
  for (IdbLayerShape* layer_shape : _layer_shape_list) {
    IdbRect rect = layer_shape->get_bounding_box();
    bounding_box_ll_x = std::min(bounding_box_ll_x, rect.get_low_x());
    bounding_box_ll_y = std::min(bounding_box_ll_y, rect.get_low_y());
    bounding_box_ur_x = std::max(bounding_box_ur_x, rect.get_high_x());
    bounding_box_ur_y = std::max(bounding_box_ur_y, rect.get_high_y());
  }

  return IdbRect(bounding_box_ll_x, bounding_box_ll_y, bounding_box_ur_x, bounding_box_ur_y);
}

IdbRect IdbPort::get_transform_bounding_box(IdbCoordinate<int32_t>* coordinate, IdbOrientTransform* db_transform)
{
  IdbRect rect = get_bounding_box();
  rect.moveByStep(coordinate->get_x(), coordinate->get_y());
  db_transform->transformRect(&rect);
  return rect;
}

void IdbPort::set_orient(IdbOrient orient)
{
  _orient = orient;
  //   set_bounding_box();
  //   set_pin_list_coodinate();
  //   set_halo_coodinate();
}

void IdbPort::set_orient_by_enum(int32_t lef_orient)
{
  _orient = IdbEnum::GetInstance()->get_site_property()->get_orient_idb_value(lef_orient);
}

void IdbPort::set_coordinate(int32_t x, int32_t y)
{
  _coordinate->set_xy(x, y);
  set_io_average_coordinate(x, y);
  set_io_bounding_box();
}

void IdbPort::set_io_average_coordinate(int32_t x, int32_t y)
{
  _io_average_coordinate->set_xy(x, y);
}

void IdbPort::set_io_bounding_box()
{
  int32_t bounding_box_ll_x = INT_MAX;
  int32_t bounding_box_ll_y = INT_MAX;
  int32_t bounding_box_ur_x = INT_MIN;
  int32_t bounding_box_ur_y = INT_MIN;
  for (IdbLayerShape* layer_shape : _layer_shape_list) {
    IdbRect rect = layer_shape->get_bounding_box();
    bounding_box_ll_x = std::min(bounding_box_ll_x, rect.get_low_x());
    bounding_box_ll_y = std::min(bounding_box_ll_y, rect.get_low_y());
    bounding_box_ur_x = std::max(bounding_box_ur_x, rect.get_high_x());
    bounding_box_ur_y = std::max(bounding_box_ur_y, rect.get_high_y());
  }

  _io_bounding_box->set_rect(_coordinate->get_x() + bounding_box_ll_x, _coordinate->get_y() + bounding_box_ll_y,
                             _coordinate->get_x() + bounding_box_ur_x, _coordinate->get_y() + bounding_box_ur_y);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbTerm::IdbTerm()
{
  _name = "";
  _direction = IdbConnectDirection::kNone;
  _type = IdbConnectType::kNone;
  _shape = IdbTermShape::kNone;
  _placement_status = IdbPlacementStatus::kNone;
  _average_position.set_x(-1);
  _average_position.set_y(-1);
  _bouding_box = new IdbRect();
  _cell_master = nullptr;
  _has_port = false;
  _is_special_net = false;
  _is_instance = false;
  // mustjoin
}

IdbTerm::~IdbTerm()
{
  for (IdbPort* port : _port_list) {
    if (port) {
      delete port;
      port = nullptr;
    }
  }

  if (_bouding_box != nullptr) {
    delete _bouding_box;
    _bouding_box = nullptr;
  }
}

void IdbTerm::set_direction(string direction)
{
  set_direction(IdbEnum::GetInstance()->get_connect_property()->get_direction(direction));
}

void IdbTerm::set_type(string type)
{
  set_type(IdbEnum::GetInstance()->get_connect_property()->get_type(type));
}

void IdbTerm::set_shape(string shape)
{
  set_shape(IdbEnum::GetInstance()->get_connect_property()->get_pin_shape(shape));
}

IdbPort* IdbTerm::add_port(IdbPort* port)
{
  IdbPort* pPort = port;
  if (pPort == nullptr) {
    pPort = new IdbPort();
  }
  _port_list.emplace_back(pPort);

  return pPort;
}

bool IdbTerm::is_multi_layer()
{
  std::set<std::string> layer_sum;
  for (IdbPort* port : _port_list) {
    for (IdbLayerShape* layer_shape : port->get_layer_shape()) {
      layer_sum.insert(layer_shape->get_layer()->get_name());
    }
  }

  return layer_sum.size() > 1 ? true : false;
}

bool IdbTerm::is_pdn()
{
  std::string name = _name;
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);

  if (0 == name.compare("VDD") || 0 == name.compare("VSS")) {
    return true;
  }

  return false;
}

bool IdbTerm::is_power()
{
  return _type == IdbConnectType::kPower;
}

bool IdbTerm::is_ground()
{
  return _type == IdbConnectType::kGround;
}

uint8_t IdbTerm::get_top_order()
{
  uint8_t top_oder = 0;
  for (IdbPort* port : _port_list) {
    top_oder = std::max(top_oder, port->findZOrderTop());
  }

  return top_oder;
}

IdbLayer* IdbTerm::get_top_layer()
{
  IdbLayer* layer = nullptr;
  for (IdbPort* port : _port_list) {
    IdbLayer* layer_top = port->get_top_layer();
    if (layer == nullptr) {
      layer = layer_top;
    } else {
      if (layer->get_order() < layer_top->get_order()) {
        layer = layer_top;
      }
    }
  }

  return layer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace idb
