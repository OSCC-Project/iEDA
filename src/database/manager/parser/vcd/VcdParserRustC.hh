/**
 * @file VcdParserRustC.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief vcd rust parser wrapper.
 * @version 0.1
 * @date 2023-10-28
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include <vector>

#include "ops/annotate_toggle_sp/AnnotateData.hh"

extern "C" {

typedef struct RustVec
{
  void* data;
  uintptr_t len;
  uintptr_t cap;
  uintptr_t type_size;
} RustVec;

typedef struct RustVCDSignal
{
  char* hash;
  char* name;
  void* bus_index;
  unsigned int signal_size;
  unsigned int signal_type;
  void* scope;
} RustVCDSignal;

typedef struct Indexes
{
  int32_t lindex;
  int32_t rindex;
} Indexes;

typedef struct RustVCDScope
{
  char* name;
  void* parent_scope;
  struct RustVec children_scope;
  struct RustVec scope_signals;
} RustVCDScope;

typedef struct RustVCDFile
{
  long long start_time;
  long long end_time;
  unsigned int time_resolution;
  unsigned int time_unit;
  char* date;
  char* version;
  char* comment;
  void* scope_root;
} RustVCDFile;

typedef struct RustTcAndSpResVecs
{
  struct RustVec signal_tc_vec;
  struct RustVec signal_duration_vec;
} RustTcAndSpResVecs;

void* rust_parse_vcd(const char* lib_path);

struct Indexes* rust_convert_signal_index(void* bus_index);
struct RustVCDFile* rust_convert_vcd_file(void* vcd_file);
struct RustVCDScope* rust_convert_vcd_scope(void* vcd_scope);
struct RustVCDSignal* rust_convert_vcd_signal(void* c_vcd_signal);
struct RustSignalTC* rust_convert_signal_tc(void* c_signal_tc);
struct RustSignalDuration* rust_convert_signal_duration(void* c_signal_duration);

struct RustTcAndSpResVecs* rust_calc_scope_tc_sp(const char* c_top_vcd_scope_name, void* c_vcd_file);
struct RustVCDScope* find_scope_by_name(const char* scope_name, void* c_vcd_file);

typedef struct RustSignalTC
{
  char* signal_name;
  uint64_t signal_tc;
} RustSignalTC;

typedef struct RustSignalDuration
{
  char* signal_name;
  uint64_t bit_0_duration;
  uint64_t bit_1_duration;
  uint64_t bit_x_duration;
  uint64_t bit_z_duration;
} RustSignalDuration;
}

namespace ipower {

/**
 * @brief The vcd reader wrapper rust parser.
 *
 */
class RustVcdReader
{
 public:
  unsigned readVcdFile(const char* vcd_file_path);

  unsigned buildAnnotateDB(const char* top_instance_name);
  unsigned calcScopeToggleAndSp(const char* top_instance_name);

  // std::vector<RustSignalTC> countTC();
  // std::vector<RustSignalDuration> countDuration();

 private:
  RustVCDFile* _vcd_file;
  void* _vcd_file_ptr;
  RustVCDScope* _top_instance_scope;

  // std::vector<RustSignalTC> _signal_tc_vec;

  std::optional<int64_t> _begin_time;  //!< simulation begin time.
  std::optional<int64_t> _end_time;    //!< simulation end time.
  AnnotateDB _annotate_db;             //!< The annotate database for store waveform data.
};

/**
 * @brief Rust C vector iterator.
 *
 * @tparam T vector element type.
 */
template <typename T>
class RustVecIterator
{
 public:
  explicit RustVecIterator(RustVec* rust_vec) : _rust_vec(rust_vec) {}
  ~RustVecIterator() = default;

  bool hasNext() { return _index < _rust_vec->len; }
  T* next()
  {
    uintptr_t ptr_move = std::is_same_v<T, void> ? _index * _rust_vec->type_size : _index;
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
#define FOREACH_VEC_ELEM(vec, T, elem) for (RustVecIterator<T> iter(vec); iter.hasNext() ? elem = iter.next(), true : false;)

/**
 * @brief Get the Rust Vec Elem object
 *
 * @tparam T
 * @param rust_vec
 * @param index
 * @return T*
 */
template <typename T>
T* GetRustVecElem(RustVec* rust_vec, uintptr_t index)
{
  uintptr_t ptr_move = std::is_same_v<T, void> ? index * rust_vec->type_size : index;
  auto* ret_value = static_cast<T*>(rust_vec->data) + ptr_move;
  return ret_value;
}
}  // namespace ipower
