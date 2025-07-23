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

#include "json.hpp"
#include "vec_layout.h"
#include "vec_net.h"
#include "vec_node.h"
#include "vec_patch.h"
#include "vec_patch_grid.h"

namespace ivec {
using json = nlohmann::ordered_json;

class VecLayoutFileIO
{
 public:
  VecLayoutFileIO(std::string dir, VecLayout* layout, VecPatchGrid* patch_grid = nullptr)
  {
    _dir = dir;
    _layout = layout;
    _patch_grid = patch_grid;
  }
  ~VecLayoutFileIO() {}

  bool saveJson();

 private:
  std::string _dir = "";
  VecLayout* _layout = nullptr;
  VecPatchGrid* _patch_grid = nullptr;

  bool saveJsonNets();
  bool saveJsonPatchs();

  void makeDir(std::string dir);
  json makeNodePair(VecNode* node1, VecNode* node2);
};

}  // namespace ivec