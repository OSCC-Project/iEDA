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
 * @File Name: contest_wrapper.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "contest_wrapper.h"

#include "IdbDesign.h"
#include "IdbLayout.h"
#include "contest_db.h"

namespace ieda_contest {

ContestWrapper::ContestWrapper(ContestDB* contest_db, idb::IdbLayout* idb_layout, idb::IdbDesign* idb_design)
{
  _contest_db = contest_db;
  _idb_layout = idb_layout;
  _idb_design = idb_design;
}

bool ContestWrapper::transfer_idb_to_contest()
{
  return true;
}

bool ContestWrapper::transfer_contest_to_idb()
{
  return true;
}

}  // namespace ieda_contest
