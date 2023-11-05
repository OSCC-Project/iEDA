#include "SpefParserRustC.hh"

#include <cstring>
#include <iostream>

namespace ista {

bool SpefRustReader::read(std::string file_path) {
  auto* rust_spef_data = rust_parser_spef(file_path.c_str());
  _spef_file =
      static_cast<RustSpefFile*>(rust_covert_spef_file(rust_spef_data));
  return true;
}

}  // namespace ista