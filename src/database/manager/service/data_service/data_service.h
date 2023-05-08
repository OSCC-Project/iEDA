#ifndef IDB_BUILDER_DATA_SERVICE
#define IDB_BUILDER_DATA_SERVICE
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
 * @file		data_service.h
 * @author		Yell
 * @copyright	(c) 2021 All Rights Reserved.
 * @date		25/05/2021
 * @version		0.1
 * @description


        This is a public db management class to provide public db interface, including read and write operation.
 *
 */

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "IdbDesign.h"
#include "def_service.h"

namespace idb {

enum class IdbDataServiceResult
{
  kServiceSuccess = 0,
  kServiceFailed
};

using std::string;
using std::vector;

class IdbDataService
{
 public:
  // creator
  IdbDataService();
  IdbDataService(IdbDefService* def_service);
  ~IdbDataService();
  IdbDataService(const IdbDataService&) = delete;
  IdbDataService& operator=(const IdbDataService&) = delete;

  // getter
  IdbDefService* get_def_service() { return _def_service; }

  string get_layout_file() { return _layout_write_folder; }
  string get_design_file() { return _design_write_folder; }

  // setter
  IdbDataServiceResult DefServiceInit(IdbDefService* def_service);
  void set_def_service(IdbDefService* def_service) { _def_service = def_service; };

  // Layout Writer(binary)
  IdbDataServiceResult LayoutFileWriteInit(const char* file_name);
  // Design Writer(binary)
  IdbDataServiceResult DesignFileWriteInit(const char* file_name);

  // Layout Reader(binary)
  IdbDataServiceResult LayoutFileReadInit(const char* file_name);
  // Design Reader(binary)
  IdbDataServiceResult DesignFileReadInit(const char* file_name);

 private:
  // IO file
  string _layout_write_folder;  // Binary layout Write
  string _design_write_folder;  // Binary design Write

  string _layout_read_folder;  // Binary layout read
  string _design_read_folder;  // Binary design read

  // db structure
  IdbDefService* _def_service;
};

}  // namespace idb

#endif  // IDB_BUILDER_DATA_SERVICE
