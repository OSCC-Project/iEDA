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
 * @file		verilog_read..h
 * @author		Yell
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
#include "string/Str.hh"
#include "verilog/VerilogParserRustC.hh"

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
class RustVerilogReader;
};  // namespace ista
class RustVerilogModule;
namespace idb {

using namespace ista;

#define kVerilogSuccess 0
#define kVerilogFail 1

#define CLOCKS_PER_MS 1000

class RustVerilogRead
{
 public:
  RustVerilogRead(IdbDefService* def_service);
  ~RustVerilogRead();

  // getter
  IdbDefService* get_service() { return _def_service; }
  bool createDb(std::string file, std::string top_module_name);
  bool createDbAutoTop(std::string file);

  IdbConnectDirection netlistToIdb(DclType port_direction) const;

  // parser
  //   int32_t parse_version(const char* version);
  //   int32_t parse_design(const char* name);
  //   int32_t parse_units(double microns);
  int32_t build_components();
  int32_t build_assign();
  int32_t build_nets();
  int32_t build_pins();

  int32_t post_process_float_io_pins();

 private:
  IdbDesign* _idb_design = nullptr;
  IdbDefService* _def_service = nullptr;
  RustVerilogReader* _rust_verilog_reader = nullptr;
  RustVerilogModule* _rust_top_module = nullptr;
};

}  // namespace idb
