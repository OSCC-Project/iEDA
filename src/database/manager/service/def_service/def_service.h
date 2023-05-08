#ifndef IDB_BUILDER_DEF_SERVICE
#define IDB_BUILDER_DEF_SERVICE
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
 * @file		def_service.h
 * @author		Yell
 * @copyright	(c) 2021 All Rights Reserved.
 * @date		25/05/2021
 * @version		0.1
 * @description


        This is a def db management class to provide def db interface, including read and write operation.
 *
 */

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "IdbDesign.h"
#include "lef_service.h"

namespace idb {

enum class IdbDefServiceResult
{
  kServiceSuccess = 0,
  kServiceFailed
};

using std::string;
using std::vector;

class IdbDefService
{
 public:
  // creator
  IdbDefService(IdbLayout* layout);
  ~IdbDefService();
  IdbDefService(const IdbDefService&) = delete;
  IdbDefService& operator=(const IdbDefService&) = delete;

  // getter
  IdbDesign* get_design();
  IdbLayout* get_layout() { return _layout; }

  string get_def_file() { return _def_file; }
  string get_def_write_file() { return _def_write_file; }

  // setter
  void set_layout(IdbLayout* layout) { _layout = layout; }

  // DEF Reader
  IdbDefServiceResult DefFileInit(const char* file_name);
  // DEF Writer
  IdbDefServiceResult DefFileWriteInit(const char* file_name);
  // Verilog Reader
  IdbDefServiceResult VerilogFileInit(const char* file_name);

 private:
  // IO file
  string _def_file;      // Read
  string _verilog_file;  // Read
  vector<FILE*> _lef_files;
  string _def_write_file;  // Write

  // db structure
  std::unique_ptr<IdbDesign> _design;  //!< changeable data package.
  IdbLayout* _layout;
};

}  // namespace idb

#endif  // IDB_BUILDER_DEF_SERVICE
