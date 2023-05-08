#ifndef IDB_BUILDER_LEF_SERVICE
#define IDB_BUILDER_LEF_SERVICE
#pragma once
/**
 * iEDA
 * Copyright (C) 2021  PCL
 *
 * This program is free software;
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @project		iDB
 * @file		lef_service.h
 * @author		Yell
 * @copyright	(c) 2021 All Rights Reserved.
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

#include "../../../data/tech/IdbCheck.h"
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
  IdbCheck* get_check();
  vector<string> get_lef_files() { return _lef_files; }

  // LEF Reader
  IdbLefServiceResult LefFileInit(vector<string> lef_files);

 private:
  // IO file
  vector<string> _lef_files;

  // <<---tbd--->>
  // db structure
  std::shared_ptr<IdbLayout> _layout;  //!< changeable data package.
  std::shared_ptr<IdbCheck> _check;    // tech add
};

}  // namespace idb

#endif  // IDB_BUILDER_LEF_SERVICE
