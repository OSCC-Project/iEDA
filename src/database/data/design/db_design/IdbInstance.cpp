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
 * @file		IdbInstance.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Component information.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbInstance.h"

#include <algorithm>

using namespace std;
namespace idb {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbInstance::IdbInstance()
{
  //_property_map = IdbEnum::GetInstance()->get_instance_property();
  _name = "";
  //   _master_name = "";
  _cell_master = nullptr;
  _pin_list = new IdbPins();

  //   _type = IdbInstanceType::kNetlist;
  _type = IdbInstanceType::kNone;
  _status = IdbPlacementStatus::kNone;
  _coordinate = new IdbCoordinate<int32_t>();
  _orient = IdbOrient::kNone;

  _weight = -1;

  _halo = nullptr;
  _route_halo = nullptr;
  _region = nullptr;
}

IdbInstance::~IdbInstance()
{
  _pin_list->reset();

  if (_coordinate) {
    delete _coordinate;
    _coordinate = nullptr;
  }

  if (_halo) {
    delete _halo;
    _halo = nullptr;
  }

  if (_route_halo) {
    delete _route_halo;
    _route_halo = nullptr;
  }

  for (auto* obs_box : _obs_box_list) {
    if (obs_box != nullptr) {
      delete obs_box;
      obs_box = nullptr;
    }
  }
  _obs_box_list.clear();
  std::vector<IdbLayerShape*>().swap(_obs_box_list);
}

void IdbInstance::set_type(string type)
{
  _type = IdbEnum::GetInstance()->get_instance_property()->get_type(type);
}

bool IdbInstance::is_flip_flop()
{
  if (_flip_flop_flag != -1) {
    return _flip_flop_flag == 1 ? true : false;
  } else {
    for (IdbPin* pin : _pin_list->get_pin_list()) {
      IdbTerm* term = pin->get_term();
      if (term->get_type() == IdbConnectType::kClock) {
        return true;
      }
    }
  }

  return false;
}

int IdbInstance::get_logic_pin_num()
{
  return _pin_list->get_net_pin_num();
}

uint IdbInstance::get_connected_pin_num()
{
  return _pin_list->get_connected_pin_num();
}

void IdbInstance::set_orient(IdbOrient orient, bool b_update)
{
  _orient = orient;
  if (b_update) {
    //   set_pin_list();
    set_bounding_box();
    set_pin_list_coodinate();
    set_halo_coodinate();
    set_obs_box_list();
  }
}

void IdbInstance::set_orient_by_enum(int32_t lef_orient)
{
  _orient = IdbEnum::GetInstance()->get_site_property()->get_orient_idb_value(lef_orient);
}

void IdbInstance::set_status_by_def_enum(int32_t status)
{
  set_status(IdbEnum::GetInstance()->get_instance_property()->get_status(status));
}
// initialize pin list while setting cell master
void IdbInstance::set_cell_master(IdbCellMaster* cell_master)
{
  _cell_master = cell_master;

  set_pin_list();

  //   set_bounding_box();
}

void IdbInstance::set_pin_list()
{
  for (IdbTerm* term : _cell_master->get_term_list()) {
    IdbPin* pin = new IdbPin();
    pin->set_pin_name(term->get_name());
    pin->set_term(term);
    pin->set_instance(this);
    // pin->set_coordinate((_coordinate->get_x()+term->get_average_position().get_x()),
    //                     (_coordinate->get_y()+term->get_average_position().get_y())) ;
    // pin->set_bounding_box();
    _pin_list->add_pin_list(pin);
  }
}

IdbPin* IdbInstance::addPin(string name)
{
  IdbPin* pin = new IdbPin();
  pin->set_pin_name(name);
  pin->set_instance(this);
  _pin_list->add_pin_list(pin);

  return pin;
}

IdbPin* IdbInstance::get_pin(string pin_name)
{
  return _pin_list->find_pin(pin_name);
}

IdbPin* IdbInstance::get_pin_by_term(string term_name)
{
  return _pin_list->find_pin_by_term(term_name);
}

int IdbInstance::get_connected_pin_number()
{
  int number = 0;
  for (auto pin : _pin_list->get_pin_list()) {
    if (pin->get_net() != nullptr) {
      ++number;
    }
  }

  return number;
}

IdbHalo* IdbInstance::set_halo(IdbHalo* halo)
{
  if (halo != nullptr) {
    _halo = halo;
  } else {
    _halo = new IdbHalo();
  }
  return _halo;
}

IdbRouteHalo* IdbInstance::set_route_halo(IdbRouteHalo* route_halo)
{
  if (route_halo != nullptr) {
    _route_halo = route_halo;
  } else {
    _route_halo = new IdbRouteHalo();
  }
  return _route_halo;
}

void IdbInstance::set_coodinate(int32_t x, int32_t y, bool b_update)
{
  _coordinate->set_xy(x, y);
  if (b_update) {
    set_bounding_box();
    set_pin_list_coodinate();
    set_halo_coodinate();
    set_obs_box_list();

    //   get_origin_coordinate();
  }
}

void IdbInstance::set_coodinate(IdbCoordinate<int32_t> coord, bool b_update)
{
  return set_coodinate(coord.get_x(), coord.get_y(), b_update);
}

void IdbInstance::set_pin_list_coodinate()
{
  //   IdbOrientTransform db_transform(_orient, _coordinate, _cell_master->get_width(), _cell_master->get_height());

  if (_cell_master != nullptr) {
    for (IdbPin* pin : _pin_list->get_pin_list()) {
      IdbTerm* term = pin->get_term();
      if (term->get_port_number() <= 0)
        continue;
      pin->set_average_coordinate((_coordinate->get_x() + term->get_average_position().get_x()),
                                  (_coordinate->get_y() + term->get_average_position().get_y()));
      pin->set_bounding_box();
      pin->set_grid_coordinate();
      //   IdbRect* pin_rect = pin->get_bounding_box();
      //   db_transform.transformRect(pin_rect);
      //   db_transform.transformCoordinate(pin->get_average_coordinate());

      if (pin->get_average_coordinate()->get_x() < get_bounding_box()->get_low_x()
          || pin->get_average_coordinate()->get_x() > get_bounding_box()->get_high_x()
          || pin->get_average_coordinate()->get_y() < get_bounding_box()->get_low_y()
          || pin->get_average_coordinate()->get_y() > get_bounding_box()->get_high_y()) {
        // std::cout << "Error pin coodinate " << std::endl;
      }
    }
  }
}

void IdbInstance::set_halo_coodinate()
{
  if (_cell_master->is_block()) {
    IdbHalo* halo = get_halo();
    if (halo != nullptr) {
      halo->set_bounding_box(get_bounding_box());
    }
  }
}

bool IdbInstance::set_bounding_box()
{
  IdbOrientTransform db_transform(_orient, _coordinate, _cell_master->get_width(), _cell_master->get_height());

  IdbRect* rect_instance = get_bounding_box();

  int32_t ll_x = _coordinate->get_x();
  int32_t ll_y = _coordinate->get_y();
  int32_t ur_x = _coordinate->get_x() + _cell_master->get_width();
  int32_t ur_y = _coordinate->get_y() + _cell_master->get_height();
  rect_instance->set_rect(ll_x, ll_y, ur_x, ur_y);
  return db_transform.transformRect(rect_instance);
}

IdbCoordinate<int32_t> IdbInstance::get_origin_coordinate()
{
  IdbOrientTransform db_transform(_orient, _coordinate, _cell_master->get_width(), _cell_master->get_height());

  int32_t x = _cell_master->get_origin_x() + _coordinate->get_x();
  int32_t y = _cell_master->get_origin_y() + _coordinate->get_y();
  IdbCoordinate<int32_t> orgin(x, y);
  db_transform.transformCoordinate(&orgin);

  return orgin;
}
/**
 * @brief
 *
 * @param coord_x coordinate x in cell master
 * @param coord_y coordinate y in cell master
 * @return IdbCoordinate<int32_t>
 */
void IdbInstance::transformCoordinate(int32_t& coord_x, int32_t& coord_y)
{
  IdbOrientTransform db_transform(_orient, _coordinate, _cell_master->get_width(), _cell_master->get_height());

  int32_t x = coord_x + _coordinate->get_x();
  int32_t y = coord_y + _coordinate->get_y();
  IdbCoordinate<int32_t> trans_coord(x, y);
  db_transform.transformCoordinate(&trans_coord);

  coord_x = trans_coord.get_x();
  coord_y = trans_coord.get_y();
}

set<IdbInstance*> IdbInstance::findNetAdjacentInstanceList()
{
  set<IdbInstance*> adjacent_instance_list;
  for (IdbPin* pin : _pin_list->get_pin_list()) {
    if (pin == nullptr || !pin->is_net_pin()) {
      continue;
    }

    for (IdbInstance* instance : pin->get_net()->get_instance_list()->get_instance_list()) {
      if (instance->get_cell_master()->is_core()) {
        adjacent_instance_list.insert(instance);
      }
    }
  }

  return adjacent_instance_list;
}

bool IdbInstance::is_io_instance()
{
  for (IdbPin* pin : _pin_list->get_pin_list()) {
    IdbNet* net = pin->get_net();
    if (net == nullptr) {
      continue;
    }
    if (net->has_io_pins()) {
      return true;
    }
  }
  return false;
}

bool IdbInstance::is_clock_instance()
{
  for (IdbPin* pin : _pin_list->get_pin_list()) {
    IdbNet* net = pin->get_net();
    if (net == nullptr) {
      continue;
    }
    if (net->is_clock()) {
      return true;
    }
  }
  return false;
}

void IdbInstance::set_obs_box_list()
{
  /// clear
  for (auto& layer_shape : _obs_box_list) {
    if (layer_shape != nullptr) {
      delete layer_shape;
      layer_shape = nullptr;
    }
  }
  _obs_box_list.clear();

  /// set obs list shape
  IdbOrientTransform db_transform(_orient, _coordinate, _cell_master->get_width(), _cell_master->get_height());
  for (IdbObs* obs : _cell_master->get_obs_list()) {
    for (IdbObsLayer* shape_layer : obs->get_obs_layer_list()) {
      IdbLayerShape* new_shape = new IdbLayerShape();
      new_shape->set_layer(shape_layer->get_shape()->get_layer());
      for (IdbRect* rect : shape_layer->get_shape()->get_rect_list()) {
        IdbRect rect_transform;
        rect_transform.set_rect(rect);
        rect_transform.moveByStep(_coordinate->get_x(), _coordinate->get_y());
        db_transform.transformRect(&rect_transform);
        new_shape->add_rect(rect_transform);
      }

      _obs_box_list.emplace_back(new_shape);
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbInstanceList::IdbInstanceList()
{
}

IdbInstanceList::~IdbInstanceList()
{
  reset();
}

void IdbInstanceList::reset(bool delete_memory)
{
  _instance_map.clear();

  for (auto* inst : _instance_list) {
    if (inst != nullptr && delete_memory) {
      delete inst;
      inst = nullptr;
    }
  }
  _instance_list.clear();
  std::vector<IdbInstance*>().swap(_instance_list);
}

IdbInstance* IdbInstanceList::find_instance(string name)
{
  auto instance = _instance_map.find(name);
  if (instance != _instance_map.end()) {
    return instance->second;
  }

  return nullptr;
}

IdbInstance* IdbInstanceList::find_instance(size_t index)
{
  if (_instance_list.size() > index) {
    return _instance_list.at(index);
  }

  return nullptr;
}

vector<IdbInstance*> IdbInstanceList::find_instance_by_master(string master_name)
{
  vector<IdbInstance*> inst_list;

  for (auto* inst : _instance_list) {
    if (inst->get_cell_master()->get_name() == master_name) {
      inst_list.push_back(inst);
    }
  }

  return inst_list;
}

IdbInstance* IdbInstanceList::add_instance(IdbInstance* instance)
{
  IdbInstance* pInstance = instance;
  if (pInstance == nullptr) {
    pInstance = new IdbInstance();
  }
  pInstance->set_id(_mutex_index++);
  _instance_list.emplace_back(pInstance);
  _instance_map.insert(make_pair(instance->get_name(), pInstance));

  return pInstance;
}

IdbInstance* IdbInstanceList::add_instance(string name)
{
  IdbInstance* pInstance = new IdbInstance();
  pInstance->set_id(_mutex_index++);
  pInstance->set_name(name);
  _instance_list.emplace_back(pInstance);
  _instance_map.insert(make_pair(name, pInstance));

  return pInstance;
}

/**
 * @Brief : remove instance by name in the instance list
 * @param  name name of instance to be removed
 * @return true
 * @return false
 */
bool IdbInstanceList::remove_instance(string name)
{
  /// remove instance from instance list map
  auto it_map = _instance_map.find(name);
  if (it_map != _instance_map.end()) {
    it_map = _instance_map.erase(it_map);
  }

  /// remove instance from instance list
  auto it = std::find_if(_instance_list.begin(), _instance_list.end(), [name](auto instance) { return name == instance->get_name(); });
  if (it == _instance_list.end()) {
    return false;
  }

  /// remove all the pin connection in net
  auto pins = (*it)->get_pin_list();
  for (auto pin : pins->get_pin_list()) {
    auto net = pin->get_net();
    if (net != nullptr) {
      net->remove_pin(pin);
    }
  }

  /// delete instance & release resource
  delete *it;
  *it = nullptr;
  _instance_list.erase(it);

  return true;
}

/**
 * @brief
 * Get pin list in the instance list by the pin name list
 *
 */
int32_t IdbInstanceList::get_pin_list_by_names(vector<string> pin_name_list, IdbPins* pin_list, IdbInstanceList* instance_list)
{
  int32_t number = 0;
  for (IdbInstance* instance : _instance_list) {
    if (instance != nullptr) {
      IdbPins* instacne_pin_list = instance->get_pin_list();
      for (IdbPin* pin : instacne_pin_list->get_pin_list()) {
        vector<string>::iterator iter = std::find(pin_name_list.begin(), pin_name_list.end(), pin->get_term_name());
        if (iter != pin_name_list.end()) {
          pin_list->add_pin_list(pin);
          instance_list->add_instance(instance);
          ++number;

          // just only one pin connected to current net in one instance
          break;
        }
      }
    }
  }

  return number;
}

int32_t IdbInstanceList::get_num(IdbInstanceType type)
{
  if (IdbInstanceType::kMax == type) {
    return _instance_list.size();
  }

  int32_t num = 0;

  for (auto inst : _instance_list) {
    if (type == IdbInstanceType::kMax || inst->get_type() == type) {
      num++;
      continue;
    }

    if (type == IdbInstanceType::kNetlist && IdbInstanceType::kNone == inst->get_type()) {
      num++;
    }
  }

  return num;
}

int32_t IdbInstanceList::get_num_by_master_type(CellMasterType type)
{
  if (CellMasterType::kMax == type) {
    return _instance_list.size();
  }

  int32_t num = 0;

  for (auto inst : _instance_list) {
    if (inst->get_cell_master()->get_type() == type) {
      num++;
    }
  }

  return num;
}

int32_t IdbInstanceList::get_num_by_master_type_range(CellMasterType type_begin, CellMasterType type_end)
{
  int32_t num = 0;

  for (auto inst : _instance_list) {
    if (inst->get_cell_master()->get_type() >= type_begin && inst->get_cell_master()->get_type() <= type_end) {
      num++;
    }
  }

  return num;
}

int32_t IdbInstanceList::get_num_core_logic()
{
  int32_t num = 0;

  for (auto inst : _instance_list) {
    if (inst->get_cell_master()->get_type() == CellMasterType::kCore && inst->get_cell_master()->is_logic()) {
      num++;
    }
  }

  return num;
}

int32_t IdbInstanceList::get_num_physics()
{
  int32_t num = 0;

  for (auto inst : _instance_list) {
    auto cell_type = inst->get_cell_master()->get_type();
    if (cell_type == CellMasterType::kCover || cell_type == CellMasterType::kCoverBump || cell_type == CellMasterType::kPadSpacer
        || (cell_type >= CellMasterType::kCoreSpacer && cell_type <= CellMasterType::kEndcapBottomRight)) {
      num++;
    }
  }

  return num;
}

uint IdbInstanceList::get_num_clockcell()
{
  uint number = 0;
  for (auto inst : _instance_list) {
    if (inst->is_clock_instance()) {
      number++;
    }
  }

  return number;
}

uint IdbInstanceList::get_connected_pin_num()
{
  uint number = 0;
  for (auto inst : _instance_list) {
    number += inst->get_connected_pin_num();
  }
  return number;
}

uint IdbInstanceList::get_iopads_pin_num()
{
  uint number = 0;
  for (auto inst : _instance_list) {
    if (inst->get_cell_master()->is_pad()) {
      number += inst->get_connected_pin_num();
    }
  }
  return number;
}

uint IdbInstanceList::get_macro_pin_num()
{
  uint number = 0;
  for (auto inst : _instance_list) {
    if (inst->get_cell_master()->is_block()) {
      number += inst->get_connected_pin_num();
    }
  }
  return number;
}

uint IdbInstanceList::get_logic_pin_num()
{
  uint number = 0;
  for (auto inst : _instance_list) {
    if (inst->get_cell_master()->get_type() == CellMasterType::kCore && inst->get_cell_master()->is_logic()) {
      number += inst->get_connected_pin_num();
    }
  }
  return number;
}

uint IdbInstanceList::get_clock_pin_num()
{
  uint number = 0;
  for (auto inst : _instance_list) {
    if (inst->is_clock_instance()) {
      number += inst->get_connected_pin_num();
    }
  }
  return number;
}

uint64_t IdbInstanceList::get_area(CellMasterType type)
{
  uint64_t inst_area = 0;
  for (auto inst : _instance_list) {
    if (type == CellMasterType::kMax || inst->get_cell_master()->get_type() == type) {
      uint64_t area = inst->get_cell_master()->get_width() * inst->get_cell_master()->get_height();
      inst_area += area;
    }
  }

  return inst_area;
}

uint64_t IdbInstanceList::get_area_by_master_type_range(CellMasterType type_begin, CellMasterType type_end)
{
  uint64_t inst_area = 0;
  for (auto inst : _instance_list) {
    if (inst->get_cell_master()->get_type() >= type_begin && inst->get_cell_master()->get_type() <= type_end) {
      uint64_t area = inst->get_cell_master()->get_width() * inst->get_cell_master()->get_height();
      inst_area += area;
    }
  }

  return inst_area;
}

uint64_t IdbInstanceList::get_area_core_logic()
{
  uint64_t inst_area = 0;
  for (auto inst : _instance_list) {
    if (inst->get_cell_master()->get_type() == CellMasterType::kCore && inst->get_cell_master()->is_logic()) {
      uint64_t area = inst->get_cell_master()->get_width() * inst->get_cell_master()->get_height();
      inst_area += area;
    }
  }

  return inst_area;
}

uint64_t IdbInstanceList::get_area_physics()
{
  uint64_t inst_area = 0;
  for (auto inst : _instance_list) {
    auto cell_type = inst->get_cell_master()->get_type();
    if (cell_type == CellMasterType::kCover || cell_type == CellMasterType::kCoverBump || cell_type == CellMasterType::kPadSpacer
        || (cell_type >= CellMasterType::kCoreSpacer && cell_type <= CellMasterType::kEndcapBottomRight)) {
      uint64_t area = inst->get_cell_master()->get_width() * inst->get_cell_master()->get_height();
      inst_area += area;
    }
  }

  return inst_area;
}

uint64_t IdbInstanceList::get_area_clock()
{
  uint64_t inst_area = 0;
  for (auto inst : _instance_list) {
    if (inst->is_clock_instance()) {
      uint64_t area = inst->get_cell_master()->get_width() * inst->get_cell_master()->get_height();
      inst_area += area;
    }
  }

  return inst_area;
}

vector<IdbInstance*> IdbInstanceList::get_iopad_list(std::vector<std::string> master_list)
{
  vector<IdbInstance*> inst_list;
  if (master_list.size() == 0) {
    for (auto inst : _instance_list) {
      if (inst->get_cell_master()->is_pad() && !inst->get_cell_master()->is_spacer()) {
        inst_list.push_back(inst);
      }
    }
  } else {
    for (auto& master_name : master_list) {
      auto insts_find = find_instance_by_master(master_name);
      inst_list.insert(inst_list.end(), std::make_move_iterator(insts_find.begin()), std::make_move_iterator(insts_find.end()));
    }
  }

  return inst_list;
}

vector<IdbInstance*> IdbInstanceList::get_corner_list(std::vector<std::string> master_list)
{
  vector<IdbInstance*> inst_list;

  if (master_list.size() == 0) {
    for (auto* inst : _instance_list) {
      auto* site = inst->get_cell_master()->get_site();
      if (site != nullptr && site->is_corner_site()) {
        inst_list.push_back(inst);
      }
    }
  } else {
    for (auto& master_name : master_list) {
      auto insts_find = find_instance_by_master(master_name);
      inst_list.insert(inst_list.end(), std::make_move_iterator(insts_find.begin()), std::make_move_iterator(insts_find.end()));
    }
  }

  return inst_list;
}

}  // namespace idb
