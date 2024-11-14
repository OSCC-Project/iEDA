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
 * @project		large model
 * @date		06/11/2024
 * @version		0.1
 * @description
 *
 */
#include <map>
#include <string>

#include "lm_layer.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace ilm {

class LmLayout
{
 public:
  LmLayout() {}
  ~LmLayout() = default;

  // getter
  LmPatchLayers& get_patch_layers() { return _patch_layers; }
  std::map<std::string, int>& get_layer_id_map() { return _layer_id_map; }
  std::map<std::string, int>& get_via_id_map() { return _via_id_map; }
  std::map<std::string, int>& get_pdn_id_map() { return _pdn_id_map; }
  std::map<std::string, int>& get_net_id_map() { return _net_id_map; }
  // setter

  // operator
  int findLayerId(std::string name);
  int findViaId(std::string name);
  int findPdnId(std::string name);
  int findNetId(std::string name);

 private:
  LmPatchLayers _patch_layers;

  std::map<std::string, int> _layer_id_map;  /// string : layer name, int : layer id begin from 1st routing layer, for example, if M1 is 1st
                                             /// routing layer, then M1 id=0, CUT1 id=1, M2 id=2 ...
  std::map<std::string, int> _via_id_map;    /// string : via name, int : via index in this map
  std::map<std::string, int> _pdn_id_map;    /// string : pdn name, int : id in the map
  std::map<std::string, int> _net_id_map;    /// string : net name, int id in the map
};

}  // namespace ilm
