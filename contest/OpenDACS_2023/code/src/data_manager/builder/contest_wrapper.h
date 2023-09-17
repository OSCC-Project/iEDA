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
 * @File Name: contest_wrapper.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace idb {
class IdbDesign;
class IdbLayout;
}  // namespace idb

namespace ieda_contest {

class ContestDB;

class ContestWrapper
{
 public:
  ContestWrapper(ContestDB* contest_db, idb::IdbLayout* idb_layout, idb::IdbDesign* idb_design);
  ~ContestWrapper() {}

  bool transfer_idb_to_contest();
  bool transfer_contest_to_idb();

 private:
  ContestDB* _contest_db = nullptr;
  idb::IdbLayout* _idb_layout = nullptr;
  idb::IdbDesign* _idb_design = nullptr;
};

}  // namespace ieda_contest