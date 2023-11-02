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
#include "VerilogParserRustC.hh"

#include "log/Log.hh"

namespace ista {

/**
 * @brief Read the verilog file use rust parser.
 *
 * @return unsigned
 */

unsigned RustVerilogReader::readVerilog(const char* verilog_file)
{
  unsigned is_ok = 1;
  LOG_INFO << "load verilog file " << verilog_file;
  // generate1
  auto* verilog_result = rust_parse_verilog(verilog_file);

  RustVerilogModule* verilog_module = nullptr;
  if (verilog_result) {
    // generate2
    verilog_module = rust_convert_raw_verilog_module(verilog_result);
    LOG_FATAL_IF(!verilog_module) << "convert verilog module failed.";
    _top_module = verilog_module;
    _verilog_modules.push_back(verilog_module);
    // generate3
    rust_free_verilog_module(verilog_result);
    LOG_INFO << "load verilog file " << verilog_file << " success.";
  } else {
    is_ok = 0;
  }

  return is_ok;
}

/**
 * @brief Link the design file to design netlist use rust parser.
 *
 */
void RustVerilogReader::linkDesign()
{
  Sta* sta = get_sta();
  const auto* const top_cell_name = _top_module->module_name;
  auto top_module_stmts = _top_module->module_stmts;
  auto port_list = _top_module->port_list;

  LOG_INFO << "link design " << top_cell_name << " start";
  _design_netlist.set_name(top_cell_name);

  void* stmt;
  FOREACH_VEC_ELEM(&top_module_stmts, void, stmt)
  {
    if (rust_is_verilog_dcls_stmt(stmt)) {
      RustVerilogDcls* verilog_dcls_struct = rust_convert_verilog_dcls(stmt);
      auto verilog_dcls = verilog_dcls_struct->verilog_dcls;
      void* verilog_dcl;
      //   FOREACH_VEC_ELEM(&verilog_dcls, void, verilog_dcl) { process_dcl_stmt(rust_convert_verilog_dcl(verilog_dcl)); }
    } else if (rust_is_module_inst_stmt(stmt)) {
      RustVerilogInst* verilog_inst = rust_convert_verilog_inst(stmt);
      const char* inst_name = verilog_inst->inst_name;
      const char* liberty_cell_name = verilog_inst->cell_name;
      auto port_connections = verilog_inst->port_connections;

      auto* inst_cell = sta->findLibertyCell(liberty_cell_name);
    }
  }
}
}  // namespace ista