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
 * @project		iDB
 * @file		lef_service.h
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
 * @description


        This is a lef db management class to provide lef db interface, including
 read and write operation.
 *
 */

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "IdbLayout.h"

namespace idb {

enum class IdbLefServiceResult
{
  kServiceSuccess = 0,
  kServiceFailed
};

using std::string;
using std::vector;

class IdbLefService
{
 public:
  IdbLefService();
  ~IdbLefService();

  IdbLefService(const IdbLefService&) = delete;
  IdbLefService& operator=(const IdbLefService&) = delete;

  // getter
  IdbLayout* get_layout();
  vector<string> get_lef_files() { return _lef_files; }

  // LEF Reader
  IdbLefServiceResult LefFileInit(vector<string> lef_files);

 private:
  // IO file
  vector<string> _lef_files;

  // <<---tbd--->>
  // db structure
  std::shared_ptr<IdbLayout> _layout;  //!< changeable data package.
};

}  // namespace idb