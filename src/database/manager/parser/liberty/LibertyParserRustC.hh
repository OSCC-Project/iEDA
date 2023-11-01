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
#include <string>
#include <type_traits>

extern "C" {

/**
 * @brief Rust C vector.
 *
 */
typedef struct RustVec {
  void* data;           //!< vec elem data storage
  uintptr_t len;        //!< vec elem num
  uintptr_t cap;        //!< vec elem capacitance
  uintptr_t type_size;  //!< vec elem type size
} RustVec;

/**
 * @brief liberty expression operation.
 *
 */
enum RustLibertyExprOp {
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
typedef struct RustLibertyExpr {
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
 * @brief Get the expr lef object
 *
 * @param c_expr
 * @return LibertyExpr*
 */
inline RustLibertyExpr* rust_get_expr_left(RustLibertyExpr* c_expr) {
  return rust_convert_expr(c_expr->left);
}

/**
 * @brief Get the expr right object
 *
 * @param c_expr
 * @return LibertyExpr*
 */
inline RustLibertyExpr* rust_get_expr_right(RustLibertyExpr* c_expr) {
  return rust_convert_expr(c_expr->right);
}

/**
 * @brief judge expr func is one.
 *
 * @param c_expr
 * @return true
 * @return false
 */
inline bool rust_expr_func_is_one(RustLibertyExpr* c_expr) {
  return c_expr->op == RustLibertyExprOp::kOne;
}

/**
 * @brief judge expr func is zero.
 *
 * @param c_expr
 * @return true
 * @return false
 */
inline bool rust_expr_func_is_zero(RustLibertyExpr* c_expr) {
  return c_expr->op == RustLibertyExprOp::kZero;
}
/**
 * @brief Rust liberty group stmt for C.
 *
 */
typedef struct RustLibertyGroupStmt {
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
typedef struct RustLibertySimpleAttrStmt {
  char* file_name;
  uintptr_t line_no;
  char* attri_name;
  const void* attri_value;
} RustLibertySimpleAttrStmt;

/**
 * @brief Rust liberty complex attribute stmt for C.
 *
 */
typedef struct RustLibertyComplexAttrStmt {
  char* file_name;
  uintptr_t line_no;
  char* attri_name;
  struct RustVec attri_values;
} RustLibertyComplexAttrStmt;

/**
 * @brief Rust liberty string value for C.
 *
 */
typedef struct RustLibertyStringValue {
  char* value;
} RustLibertyStringValue;

/**
 * @brief Rust liberty float value for C.
 *
 */
typedef struct RustLibertyFloatValue {
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
void free_c_char(char* s);

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

/**
 * @brief Rust convert simple attribute stmt to C struct.
 *
 * @param simple_attri_stmt
 * @return struct RustLibertySimpleAttrStmt*
 */
struct RustLibertySimpleAttrStmt* rust_convert_simple_attribute_stmt(
    void* simple_attri_stmt);

/**
 * @brief Rust convert complex attribute stmt to C struct.
 *
 * @param complex_attri_stmt
 * @return struct RustLibertyComplexAttrStmt*
 */
struct RustLibertyComplexAttrStmt* rust_convert_complex_attribute_stmt(
    void* complex_attri_stmt);

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
 * @brief convert Rust float attrivute
 *
 * @param float_value
 * @return struct RustLibertyFloatValue*
 */
struct RustLibertyFloatValue* rust_convert_float_value(void* float_value);
}

/**
 * @brief Rust C vector iterator.
 *
 * @tparam T vector element type.
 */
template <typename T>
class RustVecIterator {
 public:
  explicit RustVecIterator(RustVec* rust_vec) : _rust_vec(rust_vec) {}
  ~RustVecIterator() = default;

  bool hasNext() { return _index < _rust_vec->len; }
  T* next() {
    uintptr_t ptr_move =
        std::is_same_v<T, void> ? _index * _rust_vec->type_size : _index;
    auto* ret_value = static_cast<T*>(_rust_vec->data) + ptr_move;

    ++_index;
    return ret_value;
  }

 private:
  RustVec* _rust_vec;
  uintptr_t _index = 0;
};

/**
 * @brief usage:
 * RustVec* vec;
 * T* elem;
 * FOREACH_VEC_ELEM(vec, T, elem)
 * {
 *    do_something_for_elem();
 * }
 *
 */
#define FOREACH_VEC_ELEM(vec, T, elem) \
  for (RustVecIterator<T> iter(vec);   \
       iter.hasNext() ? elem = iter.next(), true : false;)

/**
 * @brief Get the Rust Vec Elem object
 *
 * @tparam T
 * @param rust_vec
 * @param index
 * @return T*
 */
template <typename T>
T* GetRustVecElem(RustVec* rust_vec, uintptr_t index) {
  uintptr_t ptr_move =
      std::is_same_v<T, void> ? index * rust_vec->type_size : index;
  auto* ret_value = static_cast<T*>(rust_vec->data) + ptr_move;
  return ret_value;
}

namespace ista {

class LibertyBuilder;

/**
 * @brief The liberty expression builder for parser function string.
 *
 */
class RustLibertyExprBuilder {
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
class RustLibertyReader {
 public:
  explicit RustLibertyReader(const char* file_name) : _file_name(file_name) {}
  ~RustLibertyReader() = default;

  RustLibertyReader(RustLibertyReader&& other) noexcept = default;
  RustLibertyReader& operator=(RustLibertyReader&& rhs) noexcept = default;

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

  void set_library_builder(LibertyBuilder* library_builder) {
    _library_builder = library_builder;
  }
  auto* get_library_builder() { return _library_builder; }

 private:
  const char* getGroupAttriName(RustLibertyGroupStmt* group);
  unsigned visitStmtInGroup(RustLibertyGroupStmt* group);

  std::string _file_name;            //!< The liberty file name.
  LibertyBuilder* _library_builder;  //!< The liberty library builder.
};

}  // namespace ista