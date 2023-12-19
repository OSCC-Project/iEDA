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
 * @File Name: feature_manager.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-0811
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

#include "IdbDesign.h"
#include "IdbLayout.h"

using namespace idb;

namespace iplf {

class FeatureManager
{
 public:
  FeatureManager(IdbLayout* idb_layout, IdbDesign* idb_design)
  {
    _idb_layout = idb_layout;
    _idb_design = idb_design;
  }
  ~FeatureManager()
  {
    _idb_layout = nullptr;
    _idb_design = nullptr;
  };

  bool save_layout(std::string path);
  bool save_instances(std::string path);
  bool save_nets(std::string path);

 private:
  IdbDesign* _idb_design = nullptr;
  IdbLayout* _idb_layout = nullptr;
};

}  // namespace iplf
