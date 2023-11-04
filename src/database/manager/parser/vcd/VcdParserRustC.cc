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

RustVCDFile* RustVcdReader::readVcdFile(const char* vcd_file_path)
{
  auto* vcd_file_ptr = rust_parse_vcd(vcd_file_path);
  auto* vcd_file = rust_convert_vcd_file(vcd_file_ptr);
  return vcd_file;
}

}  // namespace ipower