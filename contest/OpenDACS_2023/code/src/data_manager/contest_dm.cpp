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
 * @File Name: contest_dm.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "contest_dm.h"

#include "IdbDesign.h"
#include "IdbLayout.h"
#include "contest_wrapper.h"
#include "guide_parser.h"
#include "idm.h"

namespace ieda_contest {

ContestDataManager::ContestDataManager(std::string guide_input, std::string guide_output)
{
  _guide_input = guide_input;
  _guide_output = guide_output;
  _database = new ContestDB();
  _idb_layout = dmInst->get_idb_builder()->get_lef_service()->get_layout();
  _idb_design = dmInst->get_idb_builder()->get_def_service()->get_design();
}

ContestDataManager::~ContestDataManager()
{
  if (_database != nullptr) {
    delete _database;
    _database = nullptr;
  }
}

bool ContestDataManager::buildData()
{
  if (_idb_layout == nullptr || _idb_design == nullptr) {
    return false;
  }

  /// build data from idb
  ContestWrapper wrapper(_database, _idb_layout, _idb_design);
  bool b_result = wrapper.transfer_idb_to_contest();

  /// build data from guide file
  GuideParser parser;
  b_result &= parser.parse(_guide_input, _database->get_guide_nets());

  return b_result;
}

bool ContestDataManager::saveData()
{
  /// save data to idb
  if (_idb_layout == nullptr || _idb_design == nullptr) {
    return false;
  }
  ContestWrapper wrapper(_database, _idb_layout, _idb_design);
  bool b_result = wrapper.transfer_contest_to_idb();

  /// save data to guide
  GuideParser parser;
  b_result &= parser.save(_guide_output, _database->get_guide_nets());

  return b_result;
}

}  // namespace ieda_contest
