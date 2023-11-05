#pragma once
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rust-common/RustCommon.hh"

extern "C" {
void* rust_parser_spef(const char* spef_path);
void* rust_covert_spef_file(void* c_spef_data);
void* rust_convert_spef_net(void* c_spef_net);

/**
 * @brief Rust spef net data for C.
 *
 */
typedef struct RustSpefNet {
  char* _name;
  double _lcap;
  struct RustVec _conns;
  struct RustVec _caps;
  struct RustVec _ress;

} RustSpefNet;

/**
 * @brief Rust spef data file for C.
 *
 */
typedef struct RustSpefFile {
  char* _file_name;
  struct RustVec _header;
  struct RustVec _name_map;
  struct RustVec _ports;
  struct RustVec _nets;
} RustSpefFile;
}

namespace ista {

/**
 * @brief
 *
 */
class SpefRustReader {
 public:
  RustSpefFile* get_spef_file() { return _spef_file; }

  bool read(std::string file_path);
  void expand_name(unsigned num_threads);

 private:
  RustSpefFile* _spef_file;
};

}  // namespace ista