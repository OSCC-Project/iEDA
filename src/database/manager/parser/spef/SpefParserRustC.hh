#pragma once
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ista {

extern "C" {
void* rust_parser_spef(const char* spef_path);
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