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
 * @author longshy (longshy@pcl.ac.cn)
 * @brief The VerilogParser Rust C API.
 * @version 0.1
 * @date 2023-10-30
 *
 */
#include "VerilogParserRustC.hh"

#include "log/Log.hh"
#include "string/Str.hh"

namespace ista {

/**
 * @brief Read the verilog file use rust parser.
 *
 * @return unsigned
 */

unsigned RustVerilogReader::readVerilog(const char* verilog_file_path)
{
  unsigned is_ok = 1;
  LOG_INFO << "load verilog file " << verilog_file_path;
  _verilog_file_ptr = rust_parse_verilog(verilog_file_path);

  if (_verilog_file_ptr) {
    RustVerilogFile* rust_verilog_file = rust_convert_verilog_file(_verilog_file_ptr);
    auto verilog_modules = rust_verilog_file->verilog_modules;
    void* verilog_module;
    FOREACH_VEC_ELEM(&verilog_modules, void, verilog_module)
    {
      void* verilog_module_ptr = rust_convert_rc_ref_cell_module(verilog_module);
      RustVerilogModule* rust_verilog_module = rust_convert_raw_verilog_module(verilog_module_ptr);
      _verilog_modules.emplace_back(rust_verilog_module);
    }
  } else {
    is_ok = 0;
  }

  return is_ok;
}

/**
 * @brief auto set the top module without the specific module name
 * @note only support the flatten module
 */
bool RustVerilogReader::autoTopModule()
{
  LOG_INFO << "auto set top module ";
  if (_verilog_file_ptr == nullptr)
    return false;
  RustVerilogFile* rust_verilog_file = rust_convert_verilog_file(_verilog_file_ptr);
  auto verilog_modules = rust_verilog_file->verilog_modules;
  if (verilog_modules.len != 1u) {
    return false;
  }

  void* verilog_module;
  int count = 1;
  FOREACH_VEC_ELEM(&verilog_modules, void, verilog_module)
  {
    if (count-- > 0) {
      void* verilog_module_ptr = rust_convert_rc_ref_cell_module(verilog_module);
      RustVerilogModule* rust_verilog_module = rust_convert_raw_verilog_module(verilog_module_ptr);
      _top_module = rust_verilog_module;
      _top_module_name = rust_verilog_module->module_name;  // auto set the module name
    } else {
      break;
    }
  }
  return true;
}

/**
 * @brief Flatten module use rust parser.
 *
 * @param top_module_name
 * @return unsigned
 */
unsigned RustVerilogReader::flattenModule(const char* top_module_name)
{
  _top_module_name = top_module_name;
  rust_flatten_module(_verilog_file_ptr, top_module_name);
  RustVerilogFile* rust_verilog_file = rust_convert_verilog_file(_verilog_file_ptr);

  auto verilog_modules = rust_verilog_file->verilog_modules;
  void* verilog_module;
  FOREACH_VEC_ELEM(&verilog_modules, void, verilog_module)
  {
    void* verilog_module_ptr = rust_convert_rc_ref_cell_module(verilog_module);
    RustVerilogModule* rust_verilog_module = rust_convert_raw_verilog_module(verilog_module_ptr);
    if (ieda::Str::equal(rust_verilog_module->module_name, top_module_name)) {
      _top_module = rust_verilog_module;
      break;
    }
  }

  return 1;
}

}  // namespace ista