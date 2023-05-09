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
 * @file		data_service.cpp
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
* @description


        This is a data db management class to provide db interface, including read and write operation.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "data_service.h"

namespace idb {

IdbDataService::IdbDataService()
{
}

IdbDataService::IdbDataService(IdbDefService* def_service)
{
  _def_service = def_service;
}

IdbDataService::~IdbDataService()
{
}

IdbDataServiceResult IdbDataService::DefServiceInit(IdbDefService* def_service)
{
  _def_service = def_service;

  return IdbDataServiceResult::kServiceSuccess;
}

IdbDataServiceResult IdbDataService::LayoutFileWriteInit(const char* folder_name)
{
  string test_file = folder_name;
  test_file.append("/test.idb");

  FILE* file = fopen(test_file.c_str(), "wb");

  if (file == nullptr) {
    std::cout << "Can not create layout data file ( " << folder_name << " )" << std::endl;

    return IdbDataServiceResult::kServiceFailed;
  } else {
    std::cout << "Create layout data file success ( " << folder_name << " )" << std::endl;
  }

  fclose(file);

  _layout_write_folder = folder_name;

  return IdbDataServiceResult::kServiceSuccess;
}

IdbDataServiceResult IdbDataService::LayoutFileReadInit(const char* folder_name)
{
  string test_file = folder_name;
  test_file.append("/test.idb");

  FILE* file = fopen(test_file.c_str(), "wb");
  file = fopen(test_file.c_str(), "rb");

  if (file == nullptr) {
    std::cout << "Can not open layout data file ( " << folder_name << " )" << std::endl;

    return IdbDataServiceResult::kServiceFailed;
  } else {
    std::cout << "Open layout data file success ( " << folder_name << " )" << std::endl;
  }

  fclose(file);
  remove(test_file.c_str());

  _layout_read_folder = folder_name;

  return IdbDataServiceResult::kServiceSuccess;
}

}  // namespace idb
