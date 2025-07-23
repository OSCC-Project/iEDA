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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "vec_patch_grid.h"

#include "vec_grid_info.h"

namespace ivec {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
VecPatch* VecPatchGrid::findPatch(int node_row, int node_col)
{
  int patch_id = patchInfoInst.get_patch_id(node_row, node_col);
  auto it = _patchs.find(patch_id);
  if (it != _patchs.end()) {
    return &it->second;
  }

  return nullptr;
}

VecPatch* VecPatchGrid::findPatch(int patch_id)
{
  auto it = _patchs.find(patch_id);
  if (it != _patchs.end()) {
    return &it->second;
  }

  return nullptr;
}

VecPatchLayer* VecPatchGrid::findPatchLayer(int patch_id, int layer_id)
{
  auto* patch = findPatch(patch_id);
  if (patch != nullptr) {
    return patch->findLayer(layer_id);
  }

  return nullptr;
}

}  // namespace ivec
