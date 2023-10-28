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

extern "C" {

typedef struct RustVec {
  void *data;
  uintptr_t len;
  uintptr_t cap;
  uintptr_t type_size;
} RustVec;

typedef struct RustVCDScope {
  char *name;
  void *parent_scope;
  struct RustVec children_scope;
} RustVCDScope;

void *rust_parse_vcd(const char *lib_path);
}

namespace ipower {
/**
 * @brief The vcd reader wrapper rust parser.
 *
 */
class RustVcdReader {
 public:
  unsigned readVcdFile(const char *vcd_file_path);
};
}  // namespace ipower