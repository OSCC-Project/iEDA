#include "SpefParserRustC.hh"

#include <cstring>
#include <iostream>

namespace ista {

bool SpefRustReader::read(std::string file_path)
{
  _spef_file_data = rust_parser_spef(file_path.c_str());
  return true;
}

}  // namespace ista