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
#include <string>

#include "vec_layout.h"
#include "vec_node.h"
#include "vec_patch.h"
#include "vec_patch_grid.h"

namespace ivec {

class VecPatchInit
{
 public:
  VecPatchInit(VecLayout* layout, VecPatchGrid* patch_grid) : _layout(layout), _patch_grid(patch_grid) {}
  ~VecPatchInit() {}
  void init(bool is_placement_mode = false);

 private:
  VecLayout* _layout;
  VecPatchGrid* _patch_grid;

  void init_patch_grid();

  void initSubNet();
  void initSubNetFeature();

  void initLayoutPDN();
  void initLayoutInstance();
  void initLayoutIO();
  void initLayoutNets();

  std::map<int, std::pair<VecNode*, VecNode*>> splitWirePath(VecNode* node1, VecNode* node2);
};

}  // namespace ivec