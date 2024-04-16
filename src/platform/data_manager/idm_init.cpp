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
/**
 * @File Name: dm_init.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "idm.h"

namespace idm {
bool DataManager::initLef(vector<string> lef_path, bool b_techlef)
{
  _idb_lef_service = _idb_builder->buildLef(lef_path, b_techlef);
  _layout = get_idb_layout();

  return _idb_lef_service == nullptr ? false : true;
}

bool DataManager::initDef(string def_path)
{
  _idb_def_service = _idb_builder->buildDef(def_path);
  _design = get_idb_design();

  /// make original coordinate on (0,0)
  if (isNeedTransformByDie()) {
    /// transform
    transformByDie();
  }

  return _idb_def_service == nullptr ? false : true;
}

bool DataManager::initVerilog(string verilog_path, string top_module)
{
  _idb_def_service = _idb_builder->rustBuildVerilog(verilog_path, top_module);
  // _idb_def_service = _idb_builder->buildVerilog(verilog_path, top_module);
  _design = get_idb_design();

  return _idb_def_service == nullptr ? false : true;
}

}  // namespace idm
