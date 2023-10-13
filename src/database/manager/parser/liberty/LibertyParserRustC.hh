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
 * @brief Rust liberty group stmt for C.
 *
 */
typedef struct RustLibertyGroupStmt {
  char* file_name;
  uint32_t line_no;
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
  uint32_t line_no;
  char* attri_name;
  const void* attri_value;
} RustLibertySimpleAttrStmt;

/**
 * @brief Rust liberty complex attribute stmt for C.
 *
 */
typedef struct RustLibertyComplexAttrStmt {
  char* file_name;
  uint32_t line_no;
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
    auto* ret_value = static_cast<T*>(_rust_vec->data) + _index;

    if (std::is_same_v<T, void>) {
      _index += _rust_vec->type_size;
    } else {
      ++_index;
    }
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

namespace ista {

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

  unsigned visitVector(RustLibertyGroupStmt* group) { return 1; }
  unsigned visitPowerTable(RustLibertyGroupStmt* group) { return 1; }
  unsigned visitCurrentTable(RustLibertyGroupStmt* group) { return 1; }
  unsigned visitTable(RustLibertyGroupStmt* group) { return 1; }
  unsigned visitInternalPower(RustLibertyGroupStmt* group) { return 1; }
  unsigned visitTiming(RustLibertyGroupStmt* group) { return 1; }
  unsigned visitPin(RustLibertyGroupStmt* group) { return 1; }
  unsigned visitBus(RustLibertyGroupStmt* group) { return 1; }
  unsigned visitLeakagePower(RustLibertyGroupStmt* group) { return 1; }
  unsigned visitCell(RustLibertyGroupStmt* group) { return 1; }
  unsigned visitWireLoad(RustLibertyGroupStmt* group) { return 1; }
  unsigned visitLuTableTemplate(RustLibertyGroupStmt* group) { return 1; }
  unsigned visitType(RustLibertyGroupStmt* group) { return 1; }
  unsigned visitOutputCurrentTemplate(RustLibertyGroupStmt* group) { return 1; }
  unsigned visitLibrary(RustLibertyGroupStmt* group);

  unsigned visitGroup(RustLibertyGroupStmt* group);
  unsigned readLib();

 private:
  std::string _file_name;  //!< The verilog file name.
};

}  // namespace ista