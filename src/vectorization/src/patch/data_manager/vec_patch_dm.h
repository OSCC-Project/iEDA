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
#include "vec_net.h"
#include "vec_patch_grid.h"

namespace ivec {

class VecPatchDataManager
{
 public:
  VecPatchDataManager(VecLayout* layout) { _layout = layout; }
  ~VecPatchDataManager() {}

  VecPatchGrid& get_patch_grid() { return _patch_grid; }

  bool buildPatchData(bool is_placement_mode = false);
  bool buildPatchData(int patch_row_step, int patch_col_step, bool is_placement_mode = false);

 private:
  VecLayout* _layout = nullptr;
  VecPatchGrid _patch_grid;

  void init(bool is_placement_mode = false);
  void init(int patch_row_step, int patch_col_step, bool is_placement_mode = false);
};

}  // namespace ivec