#pragma once
#include <iostream>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rust-common/RustCommon.hh"

extern "C" {
void* rust_parser_spef(const char* spef_path);
void* rust_covert_spef_file(void* c_spef_data);
void* rust_convert_spef_net(void* c_spef_net);
void* rust_convert_spef_conn(void* c_spef_net);
void* rust_convert_spef_net_cap_res(void* c_spef_net_cap_res);
char* rust_expand_name(void* c_spef_data, uintptr_t index);
void rust_expand_all_name(void* c_spef_data);

char* rust_get_spef_cap_unit(void* c_spef_data);
char* rust_get_spef_res_unit(void* c_spef_data);

void rust_free_spef_data(void*);
void rust_free_spef_file(void*);
void rust_free_spef_net(void*);
void rust_free_spef_conn(void*);
void rust_free_spef_net_cap_res(void*);

typedef struct RustSpefCoord
{
  double _x = 0.0;
  double _y = 0.0;
} RustSpefCoord;

enum RustConnectionDirection
{
  kINPUT,
  kOUTPUT,
  kINOUT,
  kInternal
};

enum RustConnectionType
{
  kINTERNAL,
  kEXTERNAL,
};

/**
 * @brief Rust spef conn for C.
 *
 */
typedef struct RustSpefConnEntry
{
  RustConnectionType _conn_type;
  RustConnectionDirection _conn_direction;
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
typedef struct RustSpefResCap
{
  char* _node1;
  char* _node2;
  double _res_or_cap;
} RustSpefResCap;

/**
 * @brief Rust spef net data for C.
 *
 */
typedef struct RustSpefNet
{
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
typedef struct RustSpefFile
{
  char* _file_name;
  struct RustVec _header;
  struct RustVec _ports;
  struct RustVec _nets;
} RustSpefFile;
}

namespace ista {

/**
 * @brief
 *
 */
class SpefRustReader
{
 public:
  SpefRustReader() = default;
  ~SpefRustReader()
  {
    rust_free_spef_file(_spef_file);
    rust_free_spef_data(_rust_spef_file);
  }

  RustSpefFile* get_spef_file() { return _spef_file; }

  bool read(std::string file_path);

  void expandName() { rust_expand_all_name(_rust_spef_file); }
  char* getSpefCapUnit() { return rust_get_spef_cap_unit(_rust_spef_file); }
  char* getSpefResUnit() { return rust_get_spef_res_unit(_rust_spef_file); }

 private:
  void* _rust_spef_file = nullptr;     //!< The not converted spef file data.
  RustSpefFile* _spef_file = nullptr;  //!< The converted spef file data.
};

}  // namespace ista