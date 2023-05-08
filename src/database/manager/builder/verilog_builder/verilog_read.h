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
 * @file		verilog_read..h
 * @author		Yell
 * @copyright	(c) 2021 All Rights Reserved.
 * @date		17/11/2021
 * @version		0.1
 * @description


        There is a verilog builder to build data structure from .v file.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iostream>
#include <string>
#include <vector>

#include "def_service.h"
#include "verilog/VerilogReader.hh"

namespace ista {

class Sta;
class VerilogConstantExpr;
class VerilogDcl;
class VerilogDcls;
class VerilogInst;
class VerilogModule;
class VerilogNetConcatExpr;
class VerilogNetExpr;
class VerilogNetIDExpr;
class VerilogPortRefPortConnect;
class VerilogReader;
};  // namespace ista

namespace idb {

using namespace ista;

#define kVerilogSuccess 0
#define kVerilogFail 1

#define CLOCKS_PER_MS 1000

class VerilogRead
{
 public:
  VerilogRead(IdbDefService* def_service);
  ~VerilogRead();

  // getter
  IdbDefService* get_service() { return _def_service; }
  bool createDb(std::string file, std::string top_module_name);

  IdbConnectDirection netlistToIdb(ista::VerilogDcl::DclType port_direction) const;

  // parser
  //   int32_t parse_version(const char* version);
  //   int32_t parse_design(const char* name);
  //   int32_t parse_units(double microns);
  int32_t build_components();
  int32_t build_nets();
  int32_t build_pins();

 private:
  IdbDesign* _idb_design = nullptr;
  IdbDefService* _def_service = nullptr;
  ista::VerilogReader* _verilog_read = nullptr;
  ista::VerilogModule* _top_module = nullptr;
};
}  // namespace idb
