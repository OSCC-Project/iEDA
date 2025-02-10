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
 * @file LibertyParserRustC.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The libertyParser Rust C API.
 * @version 0.1
 * @date 2023-10-12
 *
 */
#pragma once

#include <iostream>
#include <set>
#include <string>
#include <type_traits>

#include "rust-common/RustCommon.hh"

extern "C" {

/**
 * @brief liberty expression operation.
 *
 */
enum RustLibertyExprOp
{
  kBuffer,
  kNot,
  kOr,
  kAnd,
  kXor,
  kOne,
  kZero,
  kPlus,
  kMult,
};

/**
 * @brief liberty expr.
 *
 */
typedef struct RustLibertyExpr
{
  enum RustLibertyExprOp op;
  struct RustLibertyExpr* left;
  struct RustLibertyExpr* right;
  char* port_name;
} RustLibertyExpr;

/**
 * @brief parse expression in rust.
 *
 * @param expr_str
 * @return void*
 */
void* rust_parse_expr(const char* expr_str);

/**
 * @brief convert expr to c expr.
 *
 * @param c_expr
 * @return struct RustLibertyExpr*
 */
RustLibertyExpr* rust_convert_expr(void* c_expr);

/**
 * free c expr after use.
 */
void rust_free_expr(struct RustLibertyExpr* c_expr);

/**
 * @brief Get the expr lef object
 *
 * @param c_expr
 * @return LibertyExpr*
 */
inline RustLibertyExpr* rust_get_expr_left(RustLibertyExpr* c_expr)
{
  return c_expr->left ? rust_convert_expr(c_expr->left) : nullptr;
}

/**
 * @brief Get the expr right object
 *
 * @param c_expr
 * @return LibertyExpr*
 */
inline RustLibertyExpr* rust_get_expr_right(RustLibertyExpr* c_expr)
{
  return c_expr->right ? rust_convert_expr(c_expr->right) : nullptr;
}

/**
 * @brief judge expr func is one.
 *
 * @param c_expr
 * @return true
 * @return false
 */
inline bool rust_expr_func_is_one(RustLibertyExpr* c_expr)
{
  return c_expr->op == RustLibertyExprOp::kOne;
}

/**
 * @brief judge expr func is zero.
 *
 * @param c_expr
 * @return true
 * @return false
 */
inline bool rust_expr_func_is_zero(RustLibertyExpr* c_expr)
{
  return c_expr->op == RustLibertyExprOp::kZero;
}
/**
 * @brief Rust liberty group stmt for C.
 *
 */
typedef struct RustLibertyGroupStmt
{
  char* file_name;
  uintptr_t line_no;
  char* group_name;
  struct RustVec attri_values;
  struct RustVec stmts;
} RustLibertyGroupStmt;

/**
 * @brief Rust liberty simple attribute stmt for C.
 *
 */
typedef struct RustLibertySimpleAttrStmt
{
  char* file_name;
  uintptr_t line_no;
  char* attri_name;
  const void* attri_value;
} RustLibertySimpleAttrStmt;

/**
 * @brief Rust liberty complex attribute stmt for C.
 *
 */
typedef struct RustLibertyComplexAttrStmt
{
  char* file_name;
  uintptr_t line_no;
  char* attri_name;
  struct RustVec attri_values;
} RustLibertyComplexAttrStmt;

/**
 * @brief Rust liberty string value for C.
 *
 */
typedef struct RustLibertyStringValue
{
  char* value;
} RustLibertyStringValue;

/**
 * @brief Rust liberty float value for C.
 *
 */
typedef struct RustLibertyFloatValue
{
  double value;
} RustLibertyFloatValue;

/**
 * @brief Rust parser lib interface.
 *
 * @param lib_path
 * @return void*
 */
void* rust_parse_lib(const char* lib_path);

/**
 * @brief rust free lib group memory after build data of lib.
 *
 * @param c_lib_group
 */
void rust_free_lib_group(void* c_lib_group);

/**
 * @brief free Rust string convert to C.
 *
 * @param s
 */
void lib_free_c_char(char* s);

/**
 * @brief judge whether stmt is simple attribute stmt.
 *
 * @param lib_stmt
 * @return true
 * @return false
 */
bool rust_is_simple_attri_stmt(void* lib_stmt);

/**
 * @brief judge whether stmt is complex attribute stmt.
 *
 * @param lib_stmt
 * @return true
 * @return false
 */
bool rust_is_complex_attri_stmt(void* lib_stmt);

/**
 * @brief judge whether stmt is attribute stmt.
 *
 * @param lib_stmt
 * @return true
 * @return false
 */
bool rust_is_attri_stmt(void* lib_stmt);

/**
 * @brief judge whether stmt is group stmt.
 *
 * @param lib_stmt
 * @return true
 * @return false
 */
bool rust_is_group_stmt(void* lib_stmt);

/**
 * @brief Rust convert raw point group stmt to C struct, while below
 * rust_convert_group_stmt convert dyn LibertyStmt.
 *
 * @param group_stmt
 * @return struct RustLibertyGroupStmt*
 */
struct RustLibertyGroupStmt* rust_convert_raw_group_stmt(void* group_stmt);

/**
 * @brief Rust convert group stmt to C struct.
 *
 * @param group_stmt
 * @return struct RustLibertyGroupStmt*
 */
struct RustLibertyGroupStmt* rust_convert_group_stmt(void* group_stmt);

void rust_free_group_stmt(struct RustLibertyGroupStmt* c_group_stmt);

/**
 * @brief Rust convert simple attribute stmt to C struct.
 *
 * @param simple_attri_stmt
 * @return struct RustLibertySimpleAttrStmt*
 */
struct RustLibertySimpleAttrStmt* rust_convert_simple_attribute_stmt(void* simple_attri_stmt);

void rust_free_simple_attribute_stmt(struct RustLibertySimpleAttrStmt* c_simple_attri_stmt);

/**
 * @brief Rust convert complex attribute stmt to C struct.
 *
 * @param complex_attri_stmt
 * @return struct RustLibertyComplexAttrStmt*
 */
struct RustLibertyComplexAttrStmt* rust_convert_complex_attribute_stmt(void* complex_attri_stmt);

void rust_free_complex_attribute_stmt(struct RustLibertyComplexAttrStmt* c_complex_attri_stmt);

/**
 * @brief Judge Rust attribue whether is float value.
 *
 * @param c_attribute_value
 * @return true
 * @return false
 */
bool rust_is_float_value(void* c_attribute_value);

/**
 * @brief Judge Rust attribute whether is string value.
 *
 * @param c_attribute_value
 * @return true
 * @return false
 */
bool rust_is_string_value(void* c_attribute_value);

/**
 * @brief convert Rust string attribute to C struct.
 *
 * @param string_value
 * @return struct RustLibertyStringValue*
 */
struct RustLibertyStringValue* rust_convert_string_value(void* string_value);

/**
 * strint value converted value should be release by the API.
 */
void rust_free_string_value(struct RustLibertyStringValue* c_string_value);

/**
 * @brief convert Rust float attrivute
 *
 * @param float_value
 * @return struct RustLibertyFloatValue*
 */
struct RustLibertyFloatValue* rust_convert_float_value(void* float_value);

void rust_free_float_value(struct RustLibertyFloatValue* c_float_value);
}

namespace ista {

class LibBuilder;

/**
 * @brief The liberty expression builder for parser function string.
 *
 */
class RustLibertyExprBuilder
{
 public:
  RustLibertyExprBuilder(const char* expr_str) : _expr_str(expr_str) {}
  ~RustLibertyExprBuilder() = default;

  void execute();
  RustLibertyExpr* get_result_expr() { return _result_expr; }

 private:
  std::string _expr_str;          //!< The expression string need to be parsed.
  RustLibertyExpr* _result_expr;  //!< The parsed expr result.
};

/**
 * @brief The liberty reader is used to read rust data.
 *
 */
class RustLibertyReader
{
 public:
  explicit RustLibertyReader(const char* file_name) : _file_name(file_name) {}
  ~RustLibertyReader() = default;

  RustLibertyReader(RustLibertyReader&& other) noexcept = default;
  RustLibertyReader& operator=(RustLibertyReader&& rhs) noexcept = default;

  void set_build_cells(std::set<std::string>& build_cells) {
    _build_cells = build_cells;
  }
  auto& get_build_cells() { return _build_cells; }
  bool isNeedBuild(std::string cell_name)
  {
    if (_build_cells.empty()) {
      return true;
    }
    return _build_cells.find(cell_name) != _build_cells.end();
  }

  unsigned visitSimpleAttri(RustLibertySimpleAttrStmt* attri);

  unsigned visitAxisOrValues(RustLibertyComplexAttrStmt* attri);
  unsigned visitComplexAttri(RustLibertyComplexAttrStmt* attri);

  unsigned visitLibrary(RustLibertyGroupStmt* group);
  unsigned visitLuTableTemplate(RustLibertyGroupStmt* group);
  unsigned visitWireLoad(RustLibertyGroupStmt* group);
  unsigned visitType(RustLibertyGroupStmt* group);
  unsigned visitOutputCurrentTemplate(RustLibertyGroupStmt* group);
  unsigned visitLeakagePower(RustLibertyGroupStmt* group);
  unsigned visitCell(RustLibertyGroupStmt* group);
  unsigned visitPin(RustLibertyGroupStmt* group);
  unsigned visitBus(RustLibertyGroupStmt* group);
  unsigned visitTiming(RustLibertyGroupStmt* group);
  unsigned visitInternalPower(RustLibertyGroupStmt* group);
  unsigned visitCurrentTable(RustLibertyGroupStmt* group);
  unsigned visitVector(RustLibertyGroupStmt* group);
  unsigned visitTable(RustLibertyGroupStmt* group);
  unsigned visitPowerTable(RustLibertyGroupStmt* group);

  unsigned visitGroup(RustLibertyGroupStmt* group);

  unsigned readLib();
  unsigned linkLib();

  void set_library_builder(LibBuilder* library_builder) { _library_builder = library_builder; }
  auto* get_library_builder() { return _library_builder; }

 private:
  const char* getGroupAttriName(RustLibertyGroupStmt* group);
  unsigned visitStmtInGroup(RustLibertyGroupStmt* group);

  void* _lib_file = nullptr;           //!< The parsered lib file.
  std::set<std::string> _build_cells;  //!< The needed cells.

  std::string _file_name;        //!< The liberty file name.
  LibBuilder* _library_builder;  //!< The liberty library builder.
};

}  // namespace ista