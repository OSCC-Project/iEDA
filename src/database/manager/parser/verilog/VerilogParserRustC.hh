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
#include <vector>

#include "rust-common/RustCommon.hh"

extern "C" {

/**
 * The wire or port declaration.
 */
typedef enum DclType
{
  KInput = 0,
  KInout = 1,
  KOutput = 2,
  KSupply0 = 3,
  KSupply1 = 4,
  KTri = 5,
  KWand = 6,
  KWire = 7,
  KWor = 8,
} DclType;

typedef struct Rc_RefCell_VerilogModule Rc_RefCell_VerilogModule;

typedef struct VerilogFile VerilogFile;

typedef struct VerilogModule VerilogModule;

/**
 * The port connection such as .port_id(net_id).
 */
typedef struct VerilogPortRefPortConnect VerilogPortRefPortConnect;
/**
 * @brief Rust verilog id for C.
 *
 */
typedef struct RustVerilogID
{
  char* id;
} RustVerilogID;

/**
 * @brief Rust verilog index id for C.
 *
 */
typedef struct RustVerilogIndexID
{
  char* id;
  char* base_id;
  int32_t index;
} RustVerilogIndexID;

/**
 * @brief Rust verilog slice id for C.
 *
 */
typedef struct RustVerilogSliceID
{
  char* id;
  char* base_id;
  int32_t range_base;
  int32_t range_max;
} RustVerilogSliceID;

typedef struct RustVerilogNetIDExpr
{
  uintptr_t line_no;
  const void* verilog_id;
} RustVerilogNetIDExpr;

typedef struct RustVerilogNetConcatExpr
{
  uintptr_t line_no;
  struct RustVec verilog_id_concat;
} RustVerilogNetConcatExpr;

typedef struct RustVerilogConstantExpr
{
  uintptr_t line_no;
  const void* verilog_id;
} RustVerilogConstantExpr;

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

typedef struct CRange
{
  bool has_value;
  int32_t start;
  int32_t end;
} CRange;

typedef struct RustVerilogDcl
{
  uintptr_t line_no;
  enum DclType dcl_type;
  char* dcl_name;
  struct CRange range;
} RustVerilogDcl;

/**
 * @brief Rust verilog dcls for C.
 *
 */
typedef struct RustVerilogDcls
{
  uintptr_t line_no;
  struct RustVec verilog_dcls;
} RustVerilogDcls;

/**
 * @brief Rust verilog inst for C.
 *
 */
typedef struct RustVerilogInst
{
  uintptr_t line_no;
  char* inst_name;
  char* cell_name;
  struct RustVec port_connections;
} RustVerilogInst;

/**
 * @brief Rust Verilog assign for C.
 *
 */
typedef struct RustVerilogAssign
{
  uintptr_t line_no;
  const void* left_net_expr;
  const void* right_net_expr;
} RustVerilogAssign;

/**
 * @brief Rust verilog port ref portConnect for C.
 *
 */
typedef struct RustVerilogPortRefPortConnect
{
  const void* port_id;
  void* net_expr;
} RustVerilogPortRefPortConnect;

typedef struct RustVerilogFile
{
  struct RustVec verilog_modules;
} RustVerilogFile;

/**
 * @brief Rust parser verilog interface.
 *
 * @param verilog_path
 * @return void*
 */
void* rust_parse_verilog(const char* verilog_path);

/**
 * @brief  Rust flateen module interface.
 *
 * @param c_verilog_file
 * @param top_module_name
 */
void rust_flatten_module(void* c_verilog_file, const char* top_module_name);

/**
 * @brief rust free verilog_file memory after build data of verilog.
 *
 * @param c_verilog_file
 */
void rust_free_verilog_file(void* c_verilog_file);

uintptr_t verilog_rust_vec_len(const struct RustVec* vec);
/**
 * @brief free Rust string convert to C.
 *
 * @param s
 */
void verilog_free_c_char(char* s);

struct RustVerilogID* rust_convert_verilog_id(void* c_verilog_virtual_base_id);

bool rust_is_id(void* c_verilog_virtual_base_id);

struct RustVerilogIndexID* rust_convert_verilog_index_id(void* c_verilog_virtual_base_id);

bool rust_is_bus_index_id(void* c_verilog_virtual_base_id);

const char* rust_get_index_name(struct RustVerilogSliceID* verilog_slice_id, uintptr_t index);

struct RustVerilogSliceID* rust_convert_verilog_slice_id(void* c_verilog_virtual_base_id);

bool rust_is_bus_slice_id(void* c_verilog_virtual_base_id);

struct RustVerilogNetIDExpr* rust_convert_verilog_net_id_expr(void* c_verilog_net_id_expr);

struct RustVerilogNetConcatExpr* rust_convert_verilog_net_concat_expr(void* c_verilog_net_concat_expr);

struct RustVerilogConstantExpr* rust_convert_verilog_constant_expr(void* c_verilog_constant_expr);

bool rust_is_id_expr(void* c_verilog_virtual_base_net_expr);

bool rust_is_concat_expr(void* c_verilog_virtual_base_net_expr);

bool rust_is_constant(void* c_verilog_virtual_base_net_expr);

///////////////////////////////////////////////////////////////////////////
/**
 * @brief Rust convert raw point verilog module to C struct.
 *
 * @param verilog_module
 * @return struct RustVerilogModule*
 */
struct RustVerilogModule* rust_convert_raw_verilog_module(void* verilog_module);

struct RustVerilogDcl* rust_convert_verilog_dcl(void* c_verilog_dcl_struct);

/**
 * @brief Rust convert verilog_dcls_struct to C struct.
 *
 * @param verilog_dcls_struct
 * @return struct RustVerilogDcls*
 */
struct RustVerilogDcls* rust_convert_verilog_dcls(void* verilog_dcls_struct);

/**
 * @brief Rust convert verilog_inst to C struct.
 *
 * @param verilog_inst
 * @return struct RustVerilogInst*
 */
struct RustVerilogInst* rust_convert_verilog_inst(void* verilog_inst);

/**
 * @brief Rust convert verilog_assign to C struct.
 *
 * @param c_verilog_assign
 * @return struct RustVerilogAssign*
 */
struct RustVerilogAssign* rust_convert_verilog_assign(void* c_verilog_assign);

/**
 * @brief Rust convert verilog_port_ref_port_connect to C struct.
 *
 * @param c_port_connect
 * @return struct RustVerilogPortRefPortConnect*
 */
struct RustVerilogPortRefPortConnect* rust_convert_verilog_port_ref_port_connect(void* c_port_connect);

/**
 * @brief judge whether stmt is module inst stmt.
 *
 * @param c_verilog_stmt
 * @return true
 * @return false
 */
bool rust_is_module_inst_stmt(void* c_verilog_stmt);

/**
 * @brief judge whether stmt is module assign stmt.
 *
 * @param c_verilog_stmt
 * @return true
 * @return false
 */
bool rust_is_module_assign_stmt(void* c_verilog_stmt);

/**
 * @brief judge whether stmt is verilog_dcl stmt.
 *
 * @param c_verilog_stmt
 * @return true
 * @return false
 */
bool rust_is_verilog_dcl_stmt(void* c_verilog_stmt);

/**
 * @brief judge whether stmt is verilog_dcls stmt.
 *
 * @param c_verilog_stmt
 * @return true
 * @return false
 */
bool rust_is_verilog_dcls_stmt(void* c_verilog_stmt);

/**
 * @brief judge whether stmt is module stmt.
 *
 * @param c_verilog_stmt
 * @return true
 * @return false
 */
bool rust_is_module_stmt(void* c_verilog_stmt);

/**
 * @brief  Rust convert verilog_file to C struct.
 *
 * @param c_verilog_file
 * @return struct RustVerilogFile*
 */
struct RustVerilogFile* rust_convert_verilog_file(void* c_verilog_file);

void* rust_convert_rc_ref_cell_module(void* c_module_ref);
}

namespace ista {

class RustVerilogReader
{
 public:
  explicit RustVerilogReader() = default;
  ~RustVerilogReader() = default;

  RustVerilogReader(RustVerilogReader&& other) noexcept = default;
  RustVerilogReader& operator=(RustVerilogReader&& rhs) noexcept = default;

  void* get_verilog_file_ptr() { return _verilog_file_ptr; }
  auto* get_top_module() { return _top_module; }
  auto& get_verilog_modules() { return _verilog_modules; }

  unsigned readVerilog(const char* verilog_file);
  bool autoTopModule();
  unsigned flattenModule(const char* top_module_name);

 private:
  void* _verilog_file_ptr;  // the parsered verilog file.
  std::string _top_module_name;
  std::vector<RustVerilogModule*> _verilog_modules;  //!< The current design parsed from verilog file.
  RustVerilogModule* _top_module = nullptr;          //!< The design top module.
};
}  // namespace ista