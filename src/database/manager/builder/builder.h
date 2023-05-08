#ifndef IDB_BUILDER
#define IDB_BUILDER
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


        This is a def db management class to provide def db interface, including
 read and write operation.
 *
 */

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "def_read.h"
#include "def_service.h"
#include "def_write.h"
#include "gds_write.h"
#include "lef_read.h"
#include "lef_service.h"
#include "verilog_read.h"
#include "verilog_write.h"

namespace idb {

enum class BuildState
{
  kBuildSuccess = 0,
  kBulidFail
};

class IdbBuilder
{
 public:
  IdbBuilder();
  ~IdbBuilder();
  // Read lef & def file
  IdbDefService* buildDef(string file);
  IdbDefService* buildDefGzip(string gzip_file);
  IdbLefService* buildLef(vector<string>& files);
  IdbDefService* buildVerilog(string file, std::string top_module_name = "asic_top");

  IdbDefService* buildDefFloorplan(string file);

  //   IdbDataService* buildData();
  //   IdbDataService* buildData(IdbDefService* def_service);

  // Write def
  bool saveDef(string file, DefWriteType type = DefWriteType::kChip);
  void saveVerilog(std::string verilog_file_name, std::set<std::string>& exclude_cell_names);
  bool saveGDSII(string file);

  // Write layout
  void saveLayout(string folder);
  // Read layout
  void loadLayout(string folder);

  IdbLefService* get_lef_service() { return _lef_service.get(); }
  IdbDefService* get_def_service() { return _def_service.get(); }
  //   IdbDataService* get_data_service() { return _data_service.get(); }

  /// operator
  void buildNet();
  void buildNetFeatureCoord(IdbNet* net);
  void buildPinFeatureCoord(IdbNet* net);
  void buildBus();

  void updateLefData();
  void updateMacro();

  /// loger
  void log();
  void logModule(string mudule, int32_t number = -1)
  {
    logSeperate();
    logNumber(mudule, number);
    // logSeperate();
  }
  void logSeperate() { std::cout << "**************************************************************" << std::endl; }
  void logNumber(string mudule, int32_t number = -1)
  {
    std::cout << mudule;
    if (number != -1) {
      std::cout << " number : " << number;
    }
    std::cout << std::endl;
  }
  void logInfo(string info, int32_t number = -1)
  {
    std::cout << info;
    if (number != -1) {
      std::cout << " number : " << number;
    }
    std::cout << std::endl;
  }

 private:
  std::shared_ptr<IdbDefService> _def_service;
  std::shared_ptr<IdbLefService> _lef_service;
  //   std::shared_ptr<IdbDataService> _data_service;
};

}  // namespace idb

#endif  // IDB_BUILDER
