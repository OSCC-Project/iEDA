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

#include "Log.hh"
#include "omp.h"
#include "usage.hh"
#include "vec_patch_dm.h"
#include "vec_patch_init.h"
#include "vec_grid_info.h"

namespace ivec {

bool VecPatchDataManager::buildPatchData()
{
  init();

  return true;
}

bool VecPatchDataManager::buildPatchData(int patch_row_step, int patch_col_step)
{
  init(patch_row_step, patch_col_step);

  return true;
}


void VecPatchDataManager::init()
{
  VecPatchInit patch_init(_layout, &_patch_grid);
  patch_init.init();
}

void VecPatchDataManager::init(int patch_row_step, int patch_col_step)
{
  VecPatchInfo::getInst(patch_row_step, patch_col_step);

  VecPatchInit patch_init(_layout, &_patch_grid);
  patch_init.init();
}


}  // namespace ivec