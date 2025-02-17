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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <map>
#include <string>
#include <vector>

#include "lm_patch.h"

namespace ilm {

class LmPatchGrid
{
 public:
  LmPatchGrid() {}
  ~LmPatchGrid() {}

  // getter
  std::map<int, LmPatch>& get_patchs() { return _patchs; }
  std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>>& get_patch_xy_map() { return _patch_xy_map; }

  // setter

  // operator
  LmPatch* findPatch(int node_row, int node_col);
  LmPatch* findPatch(int patch_id);

  LmPatchLayer* findPatchLayer(int patch_id, int layer_id);

 public:
 private:
  std::map<int, LmPatch> _patchs;  /// int : patch id
  std::map<int, std::pair<std::pair<int, int>, std::pair<int, int>>> _patch_xy_map;

};

}  // namespace ilm
