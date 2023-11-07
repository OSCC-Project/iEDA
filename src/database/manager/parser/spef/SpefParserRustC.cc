#include "SpefParserRustC.hh"

#include <cstring>
#include <iostream>

namespace ista {

/**
 * @brief read spef file.
 *
 * @param file_path
 * @return true
 * @return false
 */
bool SpefRustReader::read(std::string file_path) {
  _rust_spef_file = rust_parser_spef(file_path.c_str());
  _spef_file =
      static_cast<RustSpefFile*>(rust_covert_spef_file(_rust_spef_file));
  return true;
}

}  // namespace ista