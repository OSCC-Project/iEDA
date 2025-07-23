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
 * @project		vectorization
 * @date		06/11/2024
 * @version		0.1
 * @description
 *
 */
#include <map>
#include <string>

#include "vec_layer.h"
#include "vec_net.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace ivec {

class VecLayout
{
 public:
  VecLayout() {}
  ~VecLayout() {}

  // getter
  VecLayoutLayers& get_layout_layers() { return _layout_layers; }
  std::map<std::string, int>& get_net_name_map() { return _net_name_map; }
  VecGraph& get_graph() { return _graph; }

  // setter
  void add_layer_map(int id, std::string name);
  void add_via_map(int id, std::string name);
  void add_pdn_map(int id, std::string name);
  void add_net_map(int id, std::string name);
  void add_pin_map(int id, std::string inst_name, std::string pin_name);
  void add_instance_map(int id, std::string name);

  // operator
  int findLayerId(std::string name);
  std::string findLayerName(int id);
  int findViaId(std::string name);
  std::string findViaName(int id);
  int findPdnId(std::string name);
  std::string findPdnName(int id);
  int findNetId(std::string name);
  std::string findNetName(int id);
  int findPinId(std::string inst_name, std::string pin_name);
  std::pair<std::string, std::string> findPinName(int id);
  int findInstId(std::string name);
  std::string findInstName(int id);

 private:
  VecLayoutLayers _layout_layers;

  std::map<std::string, int> _layer_name_map;  /// string : layer name, int : layer id begin from 1st routing layer, for example, if M1 is
                                               /// 1st routing layer, then M1 id=0, CUT1 id=1, M2 id=2 ...
  std::map<int, std::string> _layer_id_map;    /// string : layer name, int : layer id begin from 1st routing layer, for example, if M1 is
                                               /// 1st routing layer, then M1 id=0, CUT1 id=1, M2 id=2 ...
  std::map<std::string, int> _via_name_map;    /// string : via name, int : via index in this map
  std::map<int, std::string> _via_id_map;      /// string : via name, int : via index in this map
  std::map<std::string, int> _pdn_name_map;    /// string : pdn name, int : id in the map
  std::map<int, std::string> _pdn_id_map;      /// string : pdn name, int : id in the map
  std::map<std::string, int> _net_name_map;    /// string : net name, int id in the map
  std::map<int, std::string> _net_id_map;      /// string : net name, int id in the map
  std::map<std::pair<std::string, std::string>, int>
      _pin_name_map;  /// std::pair<std::string, std::string> : instance name & pin name, int id in the map
  std::map<int, std::pair<std::string, std::string>>
      _pin_id_map;                            /// std::pair<std::string, std::string> : instance name & pin name, int id in the map
  std::map<std::string, int> _inst_name_map;  /// string : instance name, int id in the map
  std::map<int, std::string> _inst_id_map;    /// string : instance name, int id in the map

  VecGraph _graph;
};

}  // namespace ivec
