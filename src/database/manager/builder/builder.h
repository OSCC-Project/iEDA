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
#ifndef IDB_BUILDER
#define IDB_BUILDER
#pragma once
/**
 * @project		iDB
 * @file		def_service.h
 * @author		Yell
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
#include "json_write.h"
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
  IdbLefService* buildLef(vector<string>& files, bool b_techfile = false);
  IdbDefService* rustBuildVerilog(string file, std::string top_module_name = "asic_top");

  IdbDefService* buildDefFloorplan(string file);

  //   IdbDataService* buildData();
  //   IdbDataService* buildData(IdbDefService* def_service);

  // Write def
  bool saveDef(string file, DefWriteType type = DefWriteType::kChip);
  void saveVerilog(std::string verilog_file_name, std::set<std::string>& exclude_cell_names, bool is_add_space_for_escape_name);
  bool saveGDSII(string file);
  bool saveJSON(string file, string options);
  bool saveLef(string file);
  // Write layout
  void saveLayout(string folder);
  // Read layout
  void loadLayout(string folder);

  IdbLefService* get_lef_service() { return _lef_service; }
  IdbDefService* get_def_service() { return _def_service; }
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
  IdbDefService* _def_service = nullptr;
  IdbLefService* _lef_service = nullptr;
  //   std::shared_ptr<IdbDataService> _data_service;

  void checkNetPins();
};

}  // namespace idb

#endif  // IDB_BUILDER
