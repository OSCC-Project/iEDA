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
#include <string>
#include <vector>

#include "RTU.hpp"
namespace irt {

class EGRHelper
{
 public:
  EGRHelper() = default;
  ~EGRHelper() = default;
  // getter
  std::map<irt_int, irt_int>& get_db_to_egr_routing_layer_idx_map() { return _db_to_egr_routing_layer_idx_map; }
  std::map<std::string, irt_int>& get_routing_layer_name_idx_map() { return _routing_layer_name_idx_map; }
  std::map<irt_int, irt_int>& get_db_to_egr_cut_layer_idx_map() { return _db_to_egr_cut_layer_idx_map; }
  std::map<std::string, irt_int>& get_cut_layer_name_idx_map() { return _cut_layer_name_idx_map; }

  // setter
  void set_db_to_egr_routing_layer_idx_map(const std::map<irt_int, irt_int>& db_to_egr_routing_layer_idx_map)
  {
    _db_to_egr_routing_layer_idx_map = db_to_egr_routing_layer_idx_map;
  }
  void set_routing_layer_name_idx_map(const std::map<std::string, irt_int>& routing_layer_name_idx_map)
  {
    _routing_layer_name_idx_map = routing_layer_name_idx_map;
  }
  void set_db_to_egr_cut_layer_idx_map(const std::map<irt_int, irt_int>& db_to_egr_cut_layer_idx_map)
  {
    _db_to_egr_cut_layer_idx_map = db_to_egr_cut_layer_idx_map;
  }
  void set_cut_layer_name_idx_map(const std::map<std::string, irt_int>& cut_layer_name_idx_map)
  {
    _cut_layer_name_idx_map = cut_layer_name_idx_map;
  }

 private:
  std::map<irt_int, irt_int> _db_to_egr_routing_layer_idx_map;
  std::map<std::string, irt_int> _routing_layer_name_idx_map;
  std::map<irt_int, irt_int> _db_to_egr_cut_layer_idx_map;
  std::map<std::string, irt_int> _cut_layer_name_idx_map;
};

}  // namespace irt
