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
 * @project		iDB
 * @file		lef_service.cpp
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
* @description


        This is a lef db management class to provide db interface, including
read and write operation.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "lef_service.h"

#include <mutex>

namespace idb {

IdbLefService::IdbLefService()
{
  _layout = std::make_shared<IdbLayout>();
}

IdbLefService::~IdbLefService()
{
}

IdbLefServiceResult IdbLefService::LefFileInit(vector<string> lef_files)
{
  vector<string>::iterator it = lef_files.begin();
  for (; it != lef_files.end(); ++it) {
    string filename = *it;
    FILE* file = fopen(filename.c_str(), "r");
    if (file == nullptr) {
      std::cout << "Can not open LEF file ( " << filename.c_str() << " )" << std::endl;

      return IdbLefServiceResult::kServiceFailed;
    } else {
      //   std::cout << "Open LEF file success ( " << filename.c_str() << " )"
      //             << std::endl;
    }
  }
  // set lef files
  _lef_files.insert(_lef_files.end(), lef_files.begin(), lef_files.end());

  return IdbLefServiceResult::kServiceSuccess;
}

IdbLayout* IdbLefService::get_layout()
{
  if (!_layout) {
    _layout = std::make_shared<IdbLayout>();
  }

  return _layout.get();
}

}  // namespace idb
