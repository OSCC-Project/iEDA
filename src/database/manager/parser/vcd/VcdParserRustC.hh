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

#include "rust-common/RustCommon.hh"

extern "C" {

enum VCDVariableType {
  kVarEvent,
  kVarInteger,
  kVarParameter,
  kVarReal,
  kVarRealTime,
  kVarReg,
  kVarSupply0,
  kVarSupply1,
  kVarTime,
  kVarTri,
  kVarTriAnd,
  kVarTriOr,
  kVarTriReg,
  kVarTri0,
  kVarTri1,
  kVarWAnd,
  kVarWire,
  kVarWor,
  kDefault,
};

typedef struct RustVCDSignal {
  char* hash;
  char* name;
  void* bus_index;
  unsigned int signal_size;
  VCDVariableType signal_type;
  void* scope;
} RustVCDSignal;

typedef struct Indexes {
  int32_t lindex;
  int32_t rindex;
} Indexes;

typedef struct RustVCDScope {
  char* name;
  void* parent_scope;
  struct RustVec children_scope;
  struct RustVec scope_signals;
} RustVCDScope;

typedef struct RustVCDFile {
  long long start_time;
  long long end_time;
  unsigned int time_resolution;
  unsigned int time_unit;
  char* date;
  char* version;
  char* comment;
  void* scope_root;
} RustVCDFile;

typedef struct RustTcAndSpResVecs {
  struct RustVec signal_tc_vec;
  struct RustVec signal_duration_vec;
} RustTcAndSpResVecs;

void* rust_parse_vcd(const char* lib_path);

struct Indexes* rust_convert_signal_index(void* bus_index);
struct RustVCDFile* rust_convert_vcd_file(void* vcd_file);
void* rust_convert_rc_ref_cell_scope(void* c_vcd_ref);
void* rust_convert_rc_ref_cell_signal(void* c_vcd_ref);
struct RustVCDScope* rust_convert_vcd_scope(void* vcd_scope);
struct RustVCDSignal* rust_convert_vcd_signal(void* c_vcd_signal);
struct RustSignalTC* rust_convert_signal_tc(void* c_signal_tc);
struct RustSignalDuration* rust_convert_signal_duration(
    void* c_signal_duration);

struct RustTcAndSpResVecs* rust_calc_scope_tc_sp(
    const char* c_top_vcd_scope_name, void* c_vcd_file);
struct RustVCDScope* find_scope_by_name(const char* scope_name,
                                        void* c_vcd_file);
struct RustVCDSignal* find_signal_by_name(const char* scope_name,
                                          void* c_vcd_file);

typedef struct RustSignalTC {
  char* signal_name;
  uint64_t signal_tc;
} RustSignalTC;

typedef struct RustSignalDuration {
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
class RustVcdReader {
 public:
  void* readVcdFile(const char* vcd_file_path);
};

}  // namespace ipower
