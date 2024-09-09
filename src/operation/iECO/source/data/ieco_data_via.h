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

#include <map>
#include <vector>

#include "geometry_boost.h"

namespace idb {
class IdbViaMaster;
class IdbVia;
}  // namespace idb

namespace ieco {

/**
 * describe enclosure direction by longer metal for each layer
 * describe layer prefer direction by layer definition
 */
enum class Direction
{
  kNone,
  kNorth,
  kSouth,
  kMiddleVertical,
  kVertical,
  kWest,
  kEast,
  kMiddleHorizontal,
  kHorizontal,
  kMiddle,
  kMax
};

class EcoDataViaMaster
{
  struct ViaInfo
  {
    int rows;
    int cols;
    Direction top_direction;
    Direction bottom_direction;
    ieda_solver::GeometryBoost top_shape;     /// top enclosure shape
    ieda_solver::GeometryBoost bottom_shape;  /// bootom enclosure shape
  };

 public:
  EcoDataViaMaster(idb::IdbViaMaster* idb_via_master);
  ~EcoDataViaMaster();

  idb::IdbViaMaster* get_idb_via_master() { return _idb_via_master; }
  ViaInfo& get_info() { return _info; }

  bool isDefault();
  bool isMatchRowsCols(int row_num = 1, int col_num = 1);

 private:
  idb::IdbViaMaster* _idb_via_master;
  ViaInfo _info;

  void initViaInfo();
};

class EcoDataVia
{
 public:
  EcoDataVia(idb::IdbVia* idb_via);
  ~EcoDataVia();
  idb::IdbVia* get_idb_via() { return _idb_via; }
  bool is_top_connected_pin() { return _connected_pin_top; }
  bool is_bottom_connected_pin() { return _connected_pin_bottom; }
  ieda_solver::GeometryBoost& get_connect_top() { return _connect_top; }
  ieda_solver::GeometryBoost& get_connect_bottom() { return _connect_bottom; }
  bool is_connect_top_empty();
  bool is_connect_bottom_empty();

  void set_top_connected_pin(bool connected_pin = false) { _connected_pin_top = connected_pin; }
  void set_bottom_connected_pin(bool connected_pin = false) { _connected_pin_bottom = connected_pin; }
  bool intersectedTop(int llx, int lly, int urx, int ury);
  bool intersectedBottom(int llx, int lly, int urx, int ury);
  void addConnectedTop(int llx, int lly, int urx, int ury);
  void addConnectedBottom(int llx, int lly, int urx, int ury);

 private:
  idb::IdbVia* _idb_via = nullptr;
  bool _connected_pin_top = false;
  bool _connected_pin_bottom = false;
  ieda_solver::GeometryBoost _connect_top;     /// wires connected to this _idb_via top enclosure
  ieda_solver::GeometryBoost _connect_bottom;  /// wires connected to this _idb_via bootom enclosure
};

class EcoDataViaLayer
{
 public:
  EcoDataViaLayer();
  ~EcoDataViaLayer();
  std::string get_cut_layer() { return _cut_layer; }
  std::string get_bottom_layer() { return _bottom_layer; }
  std::string get_top_layer() { return _top_layer; }
  Direction get_prefer_direction_bottom() { return _prefer_direction_bottom; }
  Direction get_prefer_direction_top() { return _prefer_direction_top; }

  std::map<std::string, EcoDataViaMaster>& get_via_masters() { return _via_masters; }
  std::vector<EcoDataVia>& get_via_instances() { return _via_instances; }
  std::vector<EcoDataViaMaster> get_default();

  void set_prefer_direction_bottom(Direction direction) { _prefer_direction_bottom = direction; }
  void set_prefer_direction_top(Direction direction) { _prefer_direction_top = direction; }

  void addViaMaster(std::string master_name, idb::IdbViaMaster* idb_via_master);
  void addVia(idb::IdbVia* idb_via);
  void addVia(EcoDataVia eco_via);

 private:
  std::string _cut_layer;
  std::string _bottom_layer;
  std::string _top_layer;
  Direction _prefer_direction_bottom = Direction::kNone;
  Direction _prefer_direction_top = Direction::kNone;
  std::map<std::string, EcoDataViaMaster> _via_masters;  /// via master list for this cut layer
  //   std::map<std::string, EcoDataViaMaster> _used_masters;  /// via master list that has been used for this cut layer
  std::vector<EcoDataVia> _via_instances;  /// all via instances on this layer
};

}  // namespace ieco