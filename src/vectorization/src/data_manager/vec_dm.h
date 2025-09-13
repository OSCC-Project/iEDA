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

#include "vec_layout_dm.h"
#include "vec_net.h"
#include "vec_patch_dm.h"

namespace ivec {

class VecDataManager
{
 public:
  VecDataManager() {}
  ~VecDataManager()
  {
    if (patch_dm != nullptr) {
      delete patch_dm;
      patch_dm = nullptr;
    }
  }

  bool buildLayoutData();
  bool buildGraphData();
  bool buildPatternData();
  bool buildPatchData(const std::string dir);
  bool buildPatchData(const std::string dir, int patch_row_step, int patch_col_step);
  std::map<int, VecNet> getGraph(std::string path);

  bool checkData();
  void saveData(const std::string dir, bool batch_mode = true);
  bool readNetsToIDB(std::string dir);
  bool readNetsPatternToIDB(std::string path);

 public:
  VecLayoutDataManager layout_dm;
  VecPatchDataManager* patch_dm = nullptr;
};

}  // namespace ivec