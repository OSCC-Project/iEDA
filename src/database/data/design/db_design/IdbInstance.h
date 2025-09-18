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

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../IdbObject.h"
// #include "../../../basic/geometry/IdbGeometry.h"
#include "../IdbEnum.h"
#include "../IdbOrientTransform.h"
#include "../db_layout/IdbCellMaster.h"
#include "../db_layout/IdbSite.h"
#include "IdbHalo.h"
#include "IdbNet.h"
#include "IdbObs.h"
#include "IdbPins.h"

namespace idb {
class IdbRegion;

class IdbInstance : public IdbObject
{
 public:
  IdbInstance();
  ~IdbInstance();

  // getter
  std::string& get_name() { return _name; }
  // string get_master_name(){return _cell_master->get_name();}
  IdbCellMaster* get_cell_master() { return _cell_master; }
  IdbPins* get_pin_list() { return _pin_list; }
  IdbPin* get_pin(std::string pin_name);
  IdbPin* get_pin_by_term(std::string term_name);
  IdbPin* get_pin_Q()
  {
    for (IdbPin* pin : _pin_list->get_pin_list()) {
      if (pin->is_Q_output()) {
        return pin;
      }
    }

    return nullptr;
  }
  int get_connected_pin_number();

  IdbInstanceType& get_type() { return _type; }
  bool has_type() { return (_type > IdbInstanceType::kNone && _type < IdbInstanceType::kMax ? true : false); }
  bool is_physical() { return (_type == IdbInstanceType::kDist ? true : false); }
  bool is_timing() { return (_type == IdbInstanceType::kTiming ? true : false); }
  bool is_user() { return (_type == IdbInstanceType::kUser ? true : false); }
  bool is_netlist() { return ((!is_physical()) && (!is_timing()) && (!is_user()) ? true : false); }
  bool is_flip_flop();
  int32_t get_weight() { return _weight; }
  IdbRegion* get_region() { return _region; }
  IdbPlacementStatus& get_status() { return _status; }
  bool has_placed() { return (_status > IdbPlacementStatus::kNone && _status < IdbPlacementStatus::kMax ? true : false); }
  bool is_fixed() { return _status == IdbPlacementStatus::kFixed; }
  bool is_placed() { return _status == IdbPlacementStatus::kPlaced; }
  bool is_unplaced() { return _status == IdbPlacementStatus::kUnplaced; }
  bool is_cover() { return _status == IdbPlacementStatus::kCover; }
  IdbOrient& get_orient() { return _orient; }
  IdbCoordinate<int32_t>* get_coordinate() { return _coordinate; }
  IdbCoordinate<int32_t> get_origin_coordinate();
  int get_logic_pin_num();
  uint get_connected_pin_num();
  bool is_clock_instance();

  IdbHalo* get_halo() { return _halo; }
  bool has_halo() { return _halo == nullptr ? false : true; }
  IdbRouteHalo* get_route_halo() { return _route_halo; }
  bool has_route_halo() { return _route_halo == nullptr ? false : true; }

  vector<IdbLayerShape*>& get_obs_box_list() { return _obs_box_list; }

  // setter
  void set_name(string name) { _name = name; }
  // void set_cell_master_name(string name){_master_name = name;}
  void set_cell_master(IdbCellMaster* cell_master);
  void set_pin_list();
  IdbPin* addPin(string name);
  void set_type(string type);
  void set_type(IdbInstanceType type) { _type = type; }
  void set_weight(int32_t weight) { _weight = weight; }
  void set_region(IdbRegion* region) { _region = region; }
  void set_status(IdbPlacementStatus status) { _status = status; }
  void set_status_fixed() { _status = IdbPlacementStatus::kFixed; }
  void set_status_unplaced() { _status = IdbPlacementStatus::kUnplaced; }
  void set_status_coverd() { _status = IdbPlacementStatus::kCover; }
  void set_status_placed() { _status = IdbPlacementStatus::kPlaced; }
  void set_status_by_def_enum(int32_t status);
  void set_coodinate(int32_t x, int32_t y, bool b_update = true);
  void set_coodinate(IdbCoordinate<int32_t> coord, bool b_update = true);

  void set_orient(IdbOrient orient, bool b_update = true);
  void set_orient_by_enum(int32_t lef_orient);
  void set_as_flip_flop_flag() { _flip_flop_flag = 1; }

  IdbHalo* set_halo(IdbHalo* halo = nullptr);
  IdbRouteHalo* set_route_halo(IdbRouteHalo* route_halo = nullptr);

  void set_pin_list_coodinate();
  void set_halo_coodinate();

  bool set_bounding_box();

  void set_obs_box_list();

  // operator
  std::set<IdbInstance*> findNetAdjacentInstanceList();
  bool is_io_instance();

  void transformCoordinate(int32_t& coord_x, int32_t& coord_y);

 private:
  std::string _name;
  // string _master_name;
  IdbCellMaster* _cell_master;
  IdbPins* _pin_list;
  IdbInstanceType _type;
  IdbPlacementStatus _status;
  IdbCoordinate<int32_t>* _coordinate;
  IdbOrient _orient;
  // IdbSiteProperty* _site_property;

  int32_t _weight;
  //<-------tbd-------
  // mask shift

  IdbHalo* _halo;
  IdbRouteHalo* _route_halo;
  vector<IdbLayerShape*> _obs_box_list;

  // region
  // Group
  IdbRegion* _region;
  int _flip_flop_flag = -1;  /// -1:not set, 0:not flip flop, 1:flip flop
};

class IdbInstanceList
{
 public:
  IdbInstanceList();
  ~IdbInstanceList();

  // getter
  vector<IdbInstance*>& get_instance_list() { return _instance_list; }
  vector<IdbInstance*> get_iopad_list(std::vector<std::string> master_list = std::vector<std::string>{});
  vector<IdbInstance*> get_corner_list(std::vector<std::string> master_list = std::vector<std::string>{});
  int32_t get_num(IdbInstanceType type = IdbInstanceType::kMax);
  int32_t get_num_by_master_type(CellMasterType type = CellMasterType::kMax);
  int32_t get_num_by_master_type_range(CellMasterType type_begin = CellMasterType::kNone, CellMasterType type_end = CellMasterType::kMax);
  int32_t get_num_cover() { return get_num_by_master_type_range(CellMasterType::kCover, CellMasterType::kCoverBump); }
  int32_t get_num_block() { return get_num_by_master_type_range(CellMasterType::kBlock, CellMasterType::kBLockSoft); }
  int32_t get_num_core() { return get_num_by_master_type_range(CellMasterType::kCore, CellMasterType::kCoreWelltap); }
  int32_t get_num_core_logic();
  int32_t get_num_pad() { return get_num_by_master_type_range(CellMasterType::kPad, CellMasterType::kPadAreaIO); }
  int32_t get_num_endcap() { return get_num_by_master_type_range(CellMasterType::kEndcap, CellMasterType::kEndcapBottomRight); }
  int32_t get_num_ring() { return get_num_by_master_type(CellMasterType::kRing); }
  int32_t get_num_physics();
  int32_t get_num_tapcell() { return get_num_by_master_type(CellMasterType::kCoreWelltap); }

  uint get_num_clockcell();

  uint get_connected_pin_num();
  uint get_iopads_pin_num();
  uint get_macro_pin_num();
  uint get_logic_pin_num();
  uint get_clock_pin_num();

  uint64_t get_area(CellMasterType type = CellMasterType::kMax);
  uint64_t get_area_by_master_type_range(CellMasterType type_begin = CellMasterType::kNone, CellMasterType type_end = CellMasterType::kMax);
  uint64_t get_area_cover() { return get_area_by_master_type_range(CellMasterType::kCover, CellMasterType::kCoverBump); }
  uint64_t get_area_block() { return get_area_by_master_type_range(CellMasterType::kBlock, CellMasterType::kBLockSoft); }
  uint64_t get_area_core() { return get_area_by_master_type_range(CellMasterType::kCore, CellMasterType::kCoreWelltap); }
  uint64_t get_area_core_logic();
  uint64_t get_area_pad() { return get_area_by_master_type_range(CellMasterType::kPad, CellMasterType::kPadAreaIO); }
  uint64_t get_area_endcap() { return get_area_by_master_type_range(CellMasterType::kEndcap, CellMasterType::kEndcapBottomRight); }
  uint64_t get_area_tapcell() { return get_area_by_master_type_range(CellMasterType::kCoreWelltap, CellMasterType::kCoreWelltap); }
  uint64_t get_area_ring() { return get_area(CellMasterType::kRing); }
  uint64_t get_area_physics();
  uint64_t get_area_clock();

  IdbInstance* find_instance(string name);
  IdbInstance* find_instance(size_t index);
  vector<IdbInstance*> find_instance_by_master(string master_name);
  bool has_io_cell()
  {
    for (IdbInstance* instance : _instance_list) {
      if (instance->get_cell_master()->is_pad()) {
        return true;
      }
    }

    return false;
  }

  bool has_timing_cell()
  {
    for (auto inst : _instance_list) {
      if (inst->is_timing()) {
        return true;
      }
    }

    return false;
  }

  // setter
  IdbInstance* add_instance(IdbInstance* instance = nullptr);
  IdbInstance* add_instance(string name);
  bool remove_instance(string name);
  void reset(bool delete_memory = true);

  // operator
  void init(int32_t size) { _instance_list.reserve(size); }
  int32_t get_pin_list_by_names(vector<string> pin_name_list, IdbPins* pin_list, IdbInstanceList* instance_list);

 private:
  uint64_t _mutex_index = 0;
  std::vector<IdbInstance*> _instance_list;
  std::unordered_map<string, IdbInstance*> _instance_map;
};

}  // namespace idb
