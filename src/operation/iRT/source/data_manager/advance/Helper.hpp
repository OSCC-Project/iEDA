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

#include "Logger.hpp"
#include "RTHeader.hpp"
#include "Utility.hpp"
#include "builder.h"
#include "def_service.h"
#include "lef_service.h"

namespace irt {

class Helper
{
 public:
  Helper() = default;
  ~Helper() = default;
  // getter
  idb::IdbBuilder* get_idb_builder() { return _idb_builder; }
  std::string& get_design_name() { return _design_name; }
  std::vector<std::string>& get_lef_file_path_list() { return _lef_file_path_list; }
  std::string& get_def_file_path() { return _def_file_path; }
  std::map<int32_t, int32_t>& get_routing_idb_layer_id_to_idx_map() { return _routing_idb_layer_id_to_idx_map; }
  std::map<int32_t, int32_t>& get_cut_idb_layer_id_to_idx_map() { return _cut_idb_layer_id_to_idx_map; }
  std::map<std::string, int32_t>& get_routing_layer_name_to_idx_map() { return _routing_layer_name_to_idx_map; }
  std::map<std::string, int32_t>& get_cut_layer_name_to_idx_map() { return _cut_layer_name_to_idx_map; }
  std::map<std::string, ViaMasterIdx>& get_via_name_to_idx_map() { return _via_name_to_idx_map; }
  std::map<int32_t, std::vector<int32_t>>& get_cut_to_adjacent_routing_map() { return _cut_to_adjacent_routing_map; }
  // setter
  void set_idb_builder(idb::IdbBuilder* idb_builder) { _idb_builder = idb_builder; }
  void set_design_name(const std::string& design_name) { _design_name = design_name; }
  void set_lef_file_path_list(const std::vector<std::string>& lef_file_path_list) { _lef_file_path_list = lef_file_path_list; }
  void set_def_file_path(const std::string& def_file_path) { _def_file_path = def_file_path; }
  // function
  inline int32_t getRoutingLayerIdxByIDBLayerId(const int32_t idb_layer_id);
  inline int32_t getCutLayerIdxByIDBLayerId(const int32_t idb_layer_id);
  inline int32_t getRoutingLayerIdxByName(const std::string& routing_layer_name);
  inline int32_t getCutLayerIdxByName(const std::string& cut_layer_name);
  inline ViaMasterIdx getRTViaMasterIdxByName(const std::string& via_name);
  inline std::vector<int32_t> getAdjacentRoutingLayerIdxList(const int32_t cut_layer_idx);

 private:
  idb::IdbBuilder* _idb_builder;
  std::string _design_name;
  std::vector<std::string> _lef_file_path_list;
  std::string _def_file_path;
  std::map<int32_t, int32_t> _routing_idb_layer_id_to_idx_map;
  std::map<int32_t, int32_t> _cut_idb_layer_id_to_idx_map;
  std::map<std::string, int32_t> _routing_layer_name_to_idx_map;
  std::map<std::string, int32_t> _cut_layer_name_to_idx_map;
  std::map<std::string, ViaMasterIdx> _via_name_to_idx_map;
  std::map<int32_t, std::vector<int32_t>> _cut_to_adjacent_routing_map;
};

inline int32_t Helper::getRoutingLayerIdxByIDBLayerId(const int32_t idb_layer_id)
{
  int32_t routing_layer_idx = -1;
  if (RTUTIL.exist(_routing_idb_layer_id_to_idx_map, idb_layer_id)) {
    routing_layer_idx = _routing_idb_layer_id_to_idx_map[idb_layer_id];
  } else {
    RTLOG.error(Loc::current(), "The idb_layer_id ", idb_layer_id, " is not exist!");
  }
  return routing_layer_idx;
}

inline int32_t Helper::getCutLayerIdxByIDBLayerId(const int32_t idb_layer_id)
{
  int32_t cut_layer_idx = -1;
  if (RTUTIL.exist(_cut_idb_layer_id_to_idx_map, idb_layer_id)) {
    cut_layer_idx = _cut_idb_layer_id_to_idx_map[idb_layer_id];
  } else {
    RTLOG.error(Loc::current(), "The idb_layer_id ", idb_layer_id, " is not exist!");
  }
  return cut_layer_idx;
}

inline int32_t Helper::getRoutingLayerIdxByName(const std::string& routing_layer_name)
{
  int32_t routing_layer_idx = -1;
  if (RTUTIL.exist(_routing_layer_name_to_idx_map, routing_layer_name)) {
    routing_layer_idx = _routing_layer_name_to_idx_map[routing_layer_name];
  } else {
    RTLOG.error(Loc::current(), "The routing_layer_name '", routing_layer_name, "' is not exist!");
  }
  return routing_layer_idx;
}

inline int32_t Helper::getCutLayerIdxByName(const std::string& cut_layer_name)
{
  int32_t cut_layer_idx = -1;
  if (RTUTIL.exist(_cut_layer_name_to_idx_map, cut_layer_name)) {
    cut_layer_idx = _cut_layer_name_to_idx_map[cut_layer_name];
  } else {
    RTLOG.error(Loc::current(), "The cut_layer_name ", cut_layer_name, " is not exist!");
  }
  return cut_layer_idx;
}

inline ViaMasterIdx Helper::getRTViaMasterIdxByName(const std::string& via_name)
{
  ViaMasterIdx via_master_idx;
  if (RTUTIL.exist(_via_name_to_idx_map, via_name)) {
    via_master_idx = _via_name_to_idx_map[via_name];
  } else {
    RTLOG.error(Loc::current(), "The via_name ", via_name, " is not exist!");
  }
  return via_master_idx;
}

inline std::vector<int32_t> Helper::getAdjacentRoutingLayerIdxList(const int32_t cut_layer_idx)
{
  std::vector<int32_t> adjacent_routing_layer_idx;
  if (RTUTIL.exist(_cut_to_adjacent_routing_map, cut_layer_idx)) {
    adjacent_routing_layer_idx = _cut_to_adjacent_routing_map[cut_layer_idx];
  } else {
    RTLOG.error(Loc::current(), "The cut layer idx ", cut_layer_idx, " is not exist!");
  }
  return adjacent_routing_layer_idx;
}

}  // namespace irt
