/**
 * iEDA
 * Copyright (C) 2021  PCL
 *
 * This program is free software;
 *
 *
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @project		iDB
 * @file		data_service.cpp
 * @copyright	(c) 2021 All Rights Reserved.
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

  IdbDataService::IdbDataService() { }

  IdbDataService::IdbDataService(IdbDefService* def_service) { _def_service = def_service; }

  IdbDataService::~IdbDataService() { }

  IdbDataServiceResult IdbDataService::DefServiceInit(IdbDefService* def_service) {
    _def_service = def_service;

    return IdbDataServiceResult::kServiceSuccess;
  }

  IdbDataServiceResult IdbDataService::LayoutFileWriteInit(const char* folder_name) {
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

  IdbDataServiceResult IdbDataService::LayoutFileReadInit(const char* folder_name) {
    string test_file = folder_name;
    test_file.append("/test.idb");

    FILE* file = fopen(test_file.c_str(), "wb");
    file       = fopen(test_file.c_str(), "rb");

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
