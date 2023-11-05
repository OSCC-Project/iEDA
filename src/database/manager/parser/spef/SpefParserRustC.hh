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
void* rust_convert_spef_conn(void* c_spef_net);
void* rust_convert_spef_net_cap_res(void* c_spef_net_cap_res);

typedef struct RustSpefCoord {
  double _x;
  double _y;
} RustSpefCoord;

enum RustConnectionDirection {
  kINPUT,
  kOUTPUT,
  kINOUT,
};

enum RustConnectionType {
  kINTERNAL,
  kEXTERNAL,
};

/**
 * @brief Rust spef conn for C.
 *
 */
typedef struct RustSpefConnEntry {
  RustConnectionType _conn_type;
  RustConnectionType _conn_direction;
  char* _name;
  char* _driving_cell;
  double _load;
  int _layer;
  RustSpefCoord _coordinate;
  RustSpefCoord _ll_coordinate;
  RustSpefCoord _ur_coordinate;
} RustSpefConnEntry;

/**
 * @brief Rust spef res cap item for C.
 *
 */
typedef struct RustSpefResCap {
  char* _node1;
  char* _node2;
  double _res_or_cap;
} RustSpefResCap;

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