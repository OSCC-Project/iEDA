// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file VerilogParserRustC.hh
 * @author shy long (longshy@pcl.ac.cn)
 * @brief The VerilogParser Rust C API.
 * @version 0.1
 * @date 2023-10-30
 *
 */

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#include "netlist/Netlist.hh"

extern "C" {
/**
 * @brief Rust C vector.
 *
 */
typedef struct RustVec
{
  void* data;           //!< vec elem data storage
  uintptr_t len;        //!< vec elem num
  uintptr_t cap;        //!< vec elem capacitance
  uintptr_t type_size;  //!< vec elem type size
} RustVec;

/**
 * @brief Rust verilog module for C.
 *
 */
typedef struct RustVerilogModule
{
  uintptr_t line_no;
  char* module_name;
  struct RustVec port_list;
  struct RustVec module_stmts;
} RustVerilogModule;

/**
 * @brief Rust parser verilog interface.
 *
 * @param verilog_path
 * @return void*
 */
void* rust_parse_verilog(const char* verilog_path);

/**
 * @brief rust free verilog_module memory after build data of verilog.
 *
 * @param c_verilog_module
 */
void rust_free_verilog_module(void* c_verilog_module);

/**
 * @brief free Rust string convert to C.
 *
 * @param s
 */
void free_c_char(char* s);

///////////////////////////////////////////////////////////////////////////
/**
 * @brief Rust convert raw point verilog module to C struct.
 *
 * @param verilog_module
 * @return struct RustVerilogModule*
 */
struct RustVerilogModule* rust_convert_raw_verilog_module(void* verilog_module);
}

namespace ista {

class RustVerilogReader
{
 public:
  explicit RustVerilogReader(const char* file_name) : _file_name(file_name) {}
  ~RustVerilogReader() = default;

  RustVerilogReader(RustVerilogReader&& other) noexcept = default;
  RustVerilogReader& operator=(RustVerilogReader&& rhs) noexcept = default;

  unsigned readVerilog();
  // flatten.v
  // void linkDesign(const char* top_cell_name);
  void linkDesign();

 private:
  std::string _file_name;                            //!< The verilog file name.
  std::vector<RustVerilogModule*> _verilog_modules;  //!< The current design parsed from verilog file. whether need unique_ptr?
  RustVerilogModule* _top_module = nullptr;          //!< The design top module.
  Netlist _netlist;
};
}  // namespace ista