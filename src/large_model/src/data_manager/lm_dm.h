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

#include "lm_layout_dm.h"
#include "lm_net.h"
#include "lm_patch_dm.h"

namespace ilm {

class LmDataManager
{
 public:
  LmDataManager() {}
  ~LmDataManager()
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
  std::map<int, LmNet> getGraph(std::string path);

  bool checkData();
  void saveData(const std::string dir);

 public:
  LmLayoutDataManager layout_dm;
  LmPatchDataManager* patch_dm = nullptr;
};

}  // namespace ilm