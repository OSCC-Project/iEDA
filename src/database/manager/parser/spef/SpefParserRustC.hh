#pragma once
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rust-common/RustCommon.hh"

namespace ista {

extern "C" {
void* rust_parser_spef(const char* spef_path);

typedef struct RustSpefFile
{
  char* file_name;
  struct RustVec header;
  struct RustVec name_map;
  struct RustVec ports;
  struct RustVec nets;
} RustLibertyGroupStmt;
}

/**
 * @brief
 *
 */
class SpefRustReader
{
 public:
  bool read(std::string file_path);
  void expand_name(unsigned num_threads);

 private:
  void* _spef_file_data;
};

}  // namespace ista