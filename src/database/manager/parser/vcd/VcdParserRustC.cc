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
#include "VcdParserRustC.hh"

namespace ipower {

unsigned RustVcdReader::readVcdFile(const char* vcd_file_path) {
  auto* vcd_file = rust_parse_vcd(vcd_file_path);

  return 1;
}

}  // namespace ipower