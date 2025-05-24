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
#include "IdbPins.h"

#include <cmath>

#include "IdbInstance.h"
#include "IdbNet.h"

using namespace std;
namespace idb {

IdbPin::IdbPin()
{
  _pin_name = "";
  _io_term = nullptr;
  _b_new_term = false;
  _net_name = "";
  _is_io_pin = false;
  _net = nullptr;
  _special_net = nullptr;
  _instance = nullptr;
  _orient = IdbOrient::kN_R0;

  _average_coordinate = new IdbCoordinate<int32_t>(-1, -1);
  _location = new IdbCoordinate<int32_t>(-1, -1);
  _grid_coordinate = new IdbCoordinate<int32_t>(-1, -1);
}

IdbPin::~IdbPin()
{
  if (_b_new_term && _io_term != nullptr) {
    delete _io_term;
    _io_term = nullptr;
  }

  if (_average_coordinate != nullptr) {
    delete _average_coordinate;
    _average_coordinate = nullptr;
  }

  if (_location != nullptr) {
    delete _location;
    _location = nullptr;
  }

  if (_grid_coordinate != nullptr) {
    delete _grid_coordinate;
    _grid_coordinate = nullptr;
  }

  clear_port_layer_shape();
}

IdbTerm* IdbPin::set_term(IdbTerm* term)
{
  if (term == nullptr) {
    term = new IdbTerm();
    _b_new_term = true;
  }

  _io_term = term;

  return _io_term;
}

bool IdbPin::is_primary_input()
{
  if (is_io_pin()) {
    return IdbConnectDirection::kInput == _io_term->get_direction() ? true : false;
  }

  return false;
}

bool IdbPin::is_primary_output()
{
  if (is_io_pin()) {
    return IdbConnectDirection::kOutput == _io_term->get_direction() ? true : false;
  }

  return false;
}

bool IdbPin::is_flip_flop_clk()
{
  if (_instance != nullptr) {
    if (_instance->is_flip_flop() && _io_term->get_type() == IdbConnectType::kClock) {
      return true;
    }
  }

  return false;
}

bool IdbPin::is_multi_layer()
{
  return _io_term->is_multi_layer();
}

void IdbPin::set_orient_by_enum(int32_t lef_orient)
{
  _orient = IdbEnum::GetInstance()->get_site_property()->get_orient_idb_value(lef_orient);
}

bool IdbPin::set_bounding_box()
{
  set_pin_bounding_box();
  set_port_layer_shape();
  set_port_vias();
  return true;
}

void IdbPin::set_average_coordinate(int32_t x, int32_t y)
{
  _average_coordinate->set_xy(x, y);

  if (!is_io_pin()) {
    IdbOrientTransform db_transform(_instance->get_orient(), _instance->get_coordinate(), _instance->get_cell_master()->get_width(),
                                    _instance->get_cell_master()->get_height());
    db_transform.transformCoordinate(_average_coordinate);
  } else {
    IdbOrientTransform db_transform(this->get_orient(), this->get_location(), 0, 0);
    db_transform.transformCoordinate(_average_coordinate);
  }
}

/// set pin coordinate in track grid for standard cell
void IdbPin::set_grid_coordinate(int32_t x, int32_t y)
{
  //   if (!_instance->get_cell_master()->is_core() || is_io_pin() || _instance->get_coordinate()->is_negative()) {
  if (is_io_pin() || (_instance != nullptr && _instance->get_coordinate()->is_negative())) {
    return;
  }

  if (x == -1 || y == -1) {
    calculateGridCoordinate();

  } else {
    _grid_coordinate->set_xy(x, y);
  }

  //   IdbOrientTransform db_transform(_instance->get_orient(), _instance->get_coordinate(),
  //                                   _instance->get_cell_master()->get_width(),
  //                                   _instance->get_cell_master()->get_height());
  //   db_transform.transformCoordinate(_grid_coordinate);
  if (_grid_coordinate->get_x() == -1 || _grid_coordinate->get_y() == -1) {
    std::cout << "Error : no grid coordinate in this instance" << std::endl;
  }
}

int32_t getManhattanDistance(IdbCoordinate<int32_t>& a, IdbCoordinate<int32_t>& b)
{
  return std::abs(a.get_x() - b.get_x()) + std::abs(a.get_y() - b.get_y());
}

IdbLayerShape* IdbPin::get_bottom_routing_layer_shape()
{
  IdbLayerShape* bottom_layer = nullptr;

  for (IdbLayerShape* layer_shape : _layer_shape_list) {
    if (layer_shape->get_layer()->is_routing()) {
      if (bottom_layer == nullptr) {
        bottom_layer = layer_shape;
      } else {
        if (bottom_layer->get_layer()->get_order() > layer_shape->get_layer()->get_order()) {
          bottom_layer = layer_shape;
        }
      }
    }
  }

  if (bottom_layer == nullptr) {
    std::cout << "[IdbPin Error] : can not find layer shape for this Pin = " << _pin_name << std::endl;
  }

  return bottom_layer;
}

bool IdbPin::calculateGridCoordinate()
{
  /// find first routing layer
  IdbLayerShape* first_layer_shape = get_bottom_routing_layer_shape();
  if (first_layer_shape == nullptr) {
    return false;
  }

  /// get track grid
  IdbLayerRouting* routing_layer = dynamic_cast<IdbLayerRouting*>(first_layer_shape->get_layer());

  IdbTrackGrid* track_grid_prefer = routing_layer->get_prefer_track_grid();
  IdbTrackGrid* track_grid_nonprefer = routing_layer->get_nonprefer_track_grid();

  if (track_grid_prefer == nullptr || track_grid_nonprefer == nullptr) {
    /// set coordinate to shape
    for (IdbRect* rect : first_layer_shape->get_rect_list()) {
      _grid_coordinate->set_xy(rect->get_middle_point().get_x(), rect->get_middle_point().get_y());
      return true;
    }

    std::cout << "Warning : No track grid." << std::endl;
    return false;
  }

  int32_t start_x = 0;
  int32_t pitch_x = 0;
  int32_t start_y = 0;
  int32_t pitch_y = 0;

  if (track_grid_prefer->get_track()->is_track_direction_x()) {
    start_x = track_grid_prefer->get_track()->get_start();
    pitch_x = track_grid_prefer->get_track()->get_pitch();
    start_y = track_grid_nonprefer->get_track()->get_start();
    pitch_y = track_grid_nonprefer->get_track()->get_pitch();
  } else {
    start_x = track_grid_nonprefer->get_track()->get_start();
    pitch_x = track_grid_nonprefer->get_track()->get_pitch();
    start_y = track_grid_prefer->get_track()->get_start();
    pitch_y = track_grid_prefer->get_track()->get_pitch();
  }

  ///
  vector<IdbCoordinate<int32_t>> point_list;
  for (IdbRect* rect : first_layer_shape->get_rect_list()) {
    // int32_t low_index_x  = std::ceil((low_x - start_x) / pitch_x) * pitch_x + start_x;
    // int32_t high_index_x = std::floor((high_x - start_x) / pitch_x) * pitch_x + start_x;

    // int32_t low_index_y  = std::ceil((low_y - start_y) / pitch_y) * pitch_y + start_y;
    // int32_t high_index_y = std::floor((high_y - start_y) / pitch_y) * pitch_y + start_y;

    if (rect->get_low_x() < start_x || rect->get_low_y() < start_y) {
      continue;
    }

    int32_t low_index_x = (rect->get_low_x() - start_x) / pitch_x;
    int32_t high_index_x = (rect->get_high_x() - start_x) / pitch_x;
    int32_t low_index_y = (rect->get_low_y() - start_y) / pitch_y;
    int32_t high_index_y = (rect->get_high_y() - start_y) / pitch_y;

    if (low_index_x == high_index_x || low_index_y == high_index_y) {
      continue;
    }

    low_index_x = (low_index_x + 1) * pitch_x + start_x;
    high_index_x = high_index_x * pitch_x + start_x;

    low_index_y = (low_index_y + 1) * pitch_y + start_y;
    high_index_y = high_index_y * pitch_y + start_y;

    for (int i = low_index_x; i <= high_index_x; i += pitch_x) {
      for (int j = low_index_y; j <= high_index_y; j += pitch_y) {
        if (i < rect->get_low_x() || i > rect->get_high_x() || j < rect->get_low_y() || j > rect->get_high_y()) {
          std::cout << "Error pin grid coordinate, pin list empty." << std::endl;
          continue;
        }
        point_list.emplace_back(i, j);
      }
    }
  }

  if (point_list.empty()) {
    // std::cout << "Error: can not find  pin in track grid coordinate." << std::endl;

    /// if points in grid not exist, find the average point in the rect with max area
    IdbRect* rect_max_area = nullptr;
    constexpr uint64_t area = 0;
    for (IdbRect* rect : first_layer_shape->get_rect_list()) {
      rect_max_area = rect->get_area() > area ? rect : rect_max_area;
    }

    if (rect_max_area != nullptr) {
      _grid_coordinate->set_xy(rect_max_area->get_middle_point().get_x(), rect_max_area->get_middle_point().get_y());
    } else {
      _grid_coordinate->set_xy(_average_coordinate->get_x(), _average_coordinate->get_y());
      //   std::cout << "Error: can not find  pin in rect." << std::endl;
    }
  } else {
    int32_t min_distance = INT32_MAX;

    for (size_t i = 0; i < point_list.size(); i++) {
      IdbCoordinate<int32_t>& point = point_list[i];
      int32_t curr_distance = getManhattanDistance(*_average_coordinate, point);
      if (curr_distance < min_distance) {
        min_distance = curr_distance;
        _grid_coordinate->set_xy(point.get_x(), point.get_y());
      }
    }
  }

  if (_grid_coordinate->get_x() == -1 || _grid_coordinate->get_y() == -1) {
    std::cout << "Error pin grid coordinate" << get_pin_name() << std::endl;
  }

  return true;
}

void IdbPin::set_pin_bounding_box()
{
  IdbRect* rect = get_bounding_box();

  int32_t lacation_x = is_io_pin() ? _location->get_x() : _instance->get_coordinate()->get_x();
  int32_t lacation_y = is_io_pin() ? _location->get_y() : _instance->get_coordinate()->get_y();

  int32_t ll_x = lacation_x + _io_term->get_bounding_box()->get_low_x();
  int32_t ll_y = lacation_y + _io_term->get_bounding_box()->get_low_y();
  int32_t ur_x = lacation_x + _io_term->get_bounding_box()->get_high_x();
  int32_t ur_y = lacation_y + _io_term->get_bounding_box()->get_high_y();

  rect->set_rect(ll_x, ll_y, ur_x, ur_y);

  if (!is_io_pin()) {
    IdbOrientTransform db_transform(_instance->get_orient(), _instance->get_coordinate(), _instance->get_cell_master()->get_width(),
                                    _instance->get_cell_master()->get_height());
    db_transform.transformRect(rect);
  } else {
    IdbOrientTransform db_transform(this->get_orient(), this->get_location(), 0, 0);
    db_transform.transformRect(rect);
  }
}

// tbd  ---- not good design
void IdbPin::clear_port_layer_shape()
{
  for (auto& layer_shape : _layer_shape_list) {
    if (layer_shape != nullptr) {
      delete layer_shape;
      layer_shape = nullptr;
    }
  }
  _layer_shape_list.clear();
}

void IdbPin::set_port_layer_shape()
{
  if (_io_term->get_port_list().size() <= 0)
    return;
  clear_port_layer_shape();

  if (is_io_pin()) {
    if (!_io_term->is_placed())
      return;

    if (_io_term->is_port_exist()) {
      for (IdbPort* port : _io_term->get_port_list()) {
        IdbOrientTransform db_transform(port->get_orient(), port->get_coordinate(), 0, 0);
        for (IdbLayerShape* layer_shape : port->get_layer_shape()) {
          IdbLayerShape* layer_shape_transform = new IdbLayerShape();
          layer_shape_transform->set_layer(layer_shape->get_layer());
          for (IdbRect* rect : layer_shape->get_rect_list()) {
            IdbRect* rect_transform = new IdbRect(rect);
            rect_transform->moveByStep(port->get_coordinate()->get_x(), port->get_coordinate()->get_y());
            db_transform.transformRect(rect_transform);
            layer_shape_transform->add_rect(rect_transform);
          }
          _layer_shape_list.emplace_back(layer_shape_transform);
        }
      }
    } else {
      IdbOrientTransform db_transform(this->get_orient(), this->get_location(), 0, 0);
      for (IdbPort* port : _io_term->get_port_list()) {
        for (IdbLayerShape* layer_shape : port->get_layer_shape()) {
          IdbLayerShape* layer_shape_transform = new IdbLayerShape();
          layer_shape_transform->set_layer(layer_shape->get_layer());
          for (IdbRect* rect : layer_shape->get_rect_list()) {
            IdbRect* rect_transform = new IdbRect(rect);
            rect_transform->moveByStep(_location->get_x(), _location->get_y());
            db_transform.transformRect(rect_transform);
            layer_shape_transform->add_rect(rect_transform);
          }
          _layer_shape_list.emplace_back(layer_shape_transform);
        }
      }
    }
  } else {
    IdbOrientTransform db_transform(_instance->get_orient(), _instance->get_coordinate(), _instance->get_cell_master()->get_width(),
                                    _instance->get_cell_master()->get_height());
    for (IdbPort* port : _io_term->get_port_list()) {
      for (IdbLayerShape* layer_shape : port->get_layer_shape()) {
        IdbLayerShape* layer_shape_transform = new IdbLayerShape();
        layer_shape_transform->set_layer(layer_shape->get_layer());
        for (IdbRect* rect : layer_shape->get_rect_list()) {
          IdbRect* rect_transform = new IdbRect(rect);
          rect_transform->moveByStep(_instance->get_coordinate()->get_x(), _instance->get_coordinate()->get_y());
          db_transform.transformRect(rect_transform);
          layer_shape_transform->add_rect(rect_transform);
        }
        _layer_shape_list.emplace_back(layer_shape_transform);
      }
    }
  }
}

void IdbPin::set_port_vias()
{
  for_each(_via_list.begin(), _via_list.end(), [](IdbVia*& via) { delete via; });
  _via_list.clear();

  auto cloneVia = [&](IdbVia* via, int32_t x_off = 0, int32_t y_off = 0) {
    auto* pin_via = via->clone();
    auto* coord = pin_via->get_coordinate();
    coord->get_x() += x_off;
    coord->get_y() += y_off;

    if (!is_io_pin()) {
      IdbOrientTransform db_transform(_instance->get_orient(), _instance->get_coordinate(), _instance->get_cell_master()->get_width(),
                                      _instance->get_cell_master()->get_height());
      db_transform.transformCoordinate(coord);
    } else {
      IdbOrientTransform db_transform(this->get_orient(), this->get_location(), 0, 0);
      db_transform.transformCoordinate(coord);
    }

    return pin_via;
  };
  if (is_io_pin()) {
    if (!_io_term->is_placed())
      return;

    if (_io_term->is_port_exist()) {
      for (IdbPort* port : _io_term->get_port_list()) {
        for (IdbVia* via : port->get_via_list()) {
          auto* coord = port->get_coordinate();
          _via_list.push_back(cloneVia(via, coord->get_x(), coord->get_y()));
        }
      }
    } else {
      for (IdbPort* port : _io_term->get_port_list()) {
        for (IdbVia* via : port->get_via_list()) {
          _via_list.push_back(cloneVia(via, _location->get_x(), _location->get_y()));
        }
      }
    }
  } else {
    for (IdbPort* port : _io_term->get_port_list()) {
      for (IdbVia* via : port->get_via_list()) {
        _via_list.push_back(cloneVia(via, _instance->get_coordinate()->get_x(), _instance->get_coordinate()->get_y()));
      }
    }
  }
}

void IdbPin::adjustIOStripe(IdbCoordinate<int32_t>* start, IdbCoordinate<int32_t>* end)
{
  if (start->get_y() == end->get_y()) {
    // horizontal
    start->set_y(_average_coordinate->get_y());
    end->set_y(_average_coordinate->get_y());

  } else {
    /// vertical
    start->set_x(_average_coordinate->get_x());
    end->set_x(_average_coordinate->get_x());
  }
}

bool IdbPin::isIntersected(int x, int y, IdbLayer* layer)
{
  for (IdbLayerShape* layer_shape : get_port_box_list()) {
    auto is_intersected = layer_shape->isIntersected(x, y, layer);
    if (is_intersected) {
      return true;
    }
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbPins::IdbPins()
{
}

IdbPins::~IdbPins()
{
  _pin_list.clear();
}

uint32_t IdbPins::get_net_pin_num()
{
  uint32_t num = 0;
  for (auto pin : _pin_list) {
    if (pin->get_term()->is_pdn()) {
      continue;
    }

    num++;
  }

  return num;
}

uint IdbPins::get_connected_pin_num()
{
  uint32_t num = 0;
  for (auto pin : _pin_list) {
    if (pin->get_net() == nullptr) {
      continue;
    }

    num++;
  }

  return num;
}

IdbPin* IdbPins::find_pin(IdbPin* pin)
{
  for (IdbPin* pin_iter : _pin_list) {
    if (pin_iter->get_pin_name() == pin->get_pin_name() && pin->get_instance() == pin_iter->get_instance()) {
      return pin;
    }
  }

  return nullptr;
}

IdbPin* IdbPins::find_pin(string pin_name, std::string instance_name)
{
  for (IdbPin* pin : _pin_list) {
    if (pin->get_pin_name() == pin_name) {
      if (instance_name == "") {
        return pin;
      } else {
        auto* instance = pin->get_instance();
        if (instance != nullptr && instance->get_name() == instance_name) {
          return pin;
        }
      }
    }
  }

  return nullptr;
}

IdbPin* IdbPins::find_pin_by_term(string term_name)
{
  for (IdbPin* pin : _pin_list) {
    if (pin->get_term_name() == term_name) {
      return pin;
    }
  }

  return nullptr;
}

std::pair<IdbPin*, IdbRect*> IdbPins::find_pin_by_coordinate(IdbCoordinate<int32_t>* coordinate, IdbLayer* layer)
{
  std::pair<IdbPin*, IdbRect*> pin_info(nullptr, nullptr);

  for (IdbPin* pin : _pin_list) {
    if (pin != nullptr && pin->get_term()->is_placed()) {
      for (IdbLayerShape* layer_shape : pin->get_port_box_list()) {
        IdbRect* rect = layer_shape->contains(coordinate, layer);
        if (rect != nullptr) {
          pin_info.first = pin;
          pin_info.second = rect;
        }
      }
    }

    // if (pin->get_term()->is_special_net() || pin->get_term()->is_port_exist()) {
    //   num++;
    //   for (IdbPort* port : pin->get_term()->get_port_list()) {
    //     for (IdbLayerShape* layer_shape : port->get_layer_shape()) {
    //       IdbRect* rect = layer_shape->contains(coordinate, layer);
    //       if (rect != nullptr) {
    //         pin_info.first  = pin;
    //         pin_info.second = rect;
    //       }
    //     }
    //   }
    // }
    // else {
    //   IdbRect* bounding_box = pin->get_bounding_box();
    //   if (bounding_box->containPoint(coordinate)) {
    //     pin_info.first  = pin;
    //     pin_info.second = bounding_box;
    //   }
    // }
  }

  return pin_info;
}

IdbPin* IdbPins::find_pin_by_coordinate_list(vector<IdbCoordinate<int32_t>*>& coordinate_list, IdbLayer* layer)
{
  int32_t point_size = coordinate_list.size();
  if (point_size < _POINT_MAX_) {
    std::cout << "Error : size of point list should be larger than 2 to connect IO pin." << std::endl;
    return nullptr;
  }

  IdbCoordinate<int32_t>* point_start = coordinate_list[0];
  std::pair<IdbPin*, IdbRect*> pin_info_start = find_pin_by_coordinate(point_start, layer);
  if (pin_info_start.first != nullptr && pin_info_start.second != nullptr) {
    // /// adjust coordinate by pin
    // IdbRect* rect = pin_info_start.second;
    // bool change_follow = coordinate_list.size() == _POINT_MAX_ ? false : true;
    // rect->adjustCoordinate(point_start, coordinate_list[1], change_follow);
    return pin_info_start.first;
  }

  IdbCoordinate<int32_t>* point_end = coordinate_list[point_size - 1];
  std::pair<IdbPin*, IdbRect*> pin_info_end = find_pin_by_coordinate(point_end, layer);
  if (pin_info_end.first != nullptr && pin_info_end.second != nullptr) {
    // /// adjust coordinate by pin
    // IdbRect* rect = pin_info_end.second;
    // bool change_follow = coordinate_list.size() == _POINT_MAX_ ? false : true;
    // rect->adjustCoordinate(point_end, coordinate_list[point_size - 2], change_follow);
    return pin_info_end.first;
  }

  return nullptr;
}

IdbPin* IdbPins::add_pin_list(IdbPin* pin)
{
  //   IdbPin* pPin = nullptr;
  //   if (pin == nullptr) {
  //     pPin = new IdbPin();
  //     _pin_list.emplace_back(pPin);
  //   } else {
  //     pPin = find_pin(pin);
  //     if (nullptr == pPin) {
  //       _pin_list.emplace_back(pin);
  //       pPin = pin;
  //     }
  //   }
  IdbPin* pPin = nullptr;
  if (pin == nullptr) {
    pPin = new IdbPin();
    _pin_list.emplace_back(pPin);
  } else {
    _pin_list.emplace_back(pin);
    pPin = pin;
  }

  return pPin;
}

IdbPin* IdbPins::add_pin_list(string pin_name)
{
  //   IdbPin* pPin = find_pin(pin_name);
  //   if (pPin == nullptr) {
  //     pPin = new IdbPin();
  //     pPin->set_pin_name(pin_name);
  //     _pin_list.emplace_back(pPin);
  //   }
  IdbPin* pPin = new IdbPin();
  pPin->set_pin_name(pin_name);
  _pin_list.emplace_back(pPin);

  return pPin;
}
/**
 * check if same pin exist and remove it
 */
void IdbPins::checkPins()
{
  std::set<std::string> pin_name_set;
  for (auto it = _pin_list.begin(); it != _pin_list.end();) {
    size_t pin_num = pin_name_set.size();
    auto pin = *it;
    std::string name = pin->get_pin_name();
    if (pin->get_instance() != nullptr) {
      name = pin->get_instance()->get_name() + name;
    }
    pin_name_set.insert(name);
    /// if has same instance+pin
    if (pin_name_set.size() == pin_num) {
      it = _pin_list.erase(it);
      continue;
    }

    it++;
  }
}

void IdbPins::reset()
{
  for (auto* pin : _pin_list) {
    if (pin != nullptr) {
      delete pin;
      pin = nullptr;
    }
  }
  _pin_list.clear();
  std::vector<IdbPin*>().swap(_pin_list);
}

void IdbPins::remove_pin(IdbPin* pin_remove)
{
  for (auto it = _pin_list.begin(); it != _pin_list.end(); it++) {
    IdbPin* pin = *it;
    if (pin == pin_remove) {
      pin->remove_net();
      _pin_list.erase(it);

      return;
    }
  }
}

int32_t IdbPins::getIOPortWidth()
{
  int32_t width = 0;
  for (auto it = _pin_list.begin(); it != _pin_list.end(); it++) {
    IdbPin* pin = *it;
    for (IdbLayerShape* layer_shape : pin->get_port_box_list()) {
      for (IdbRect* rect : layer_shape->get_rect_list()) {
        int32_t port_width = rect->get_min_length();
        width = std::max(width, port_width);
      }
    }
  }
  return width;
}
}  // namespace idb
