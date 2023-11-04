#include <cstring>
#include <iostream>

#include "SpefParserRustC.hh"

int main() {
  //   std::string spef_file_str =
  //   "/home/immelon/projects/iPD/src/database/manager/parser/spef/spef-parser/"
  //   "aes.spef";
  std::string spef_file_str =
      "/home/taosimin/iEDA/src/database/manager/parser/spef/spef-parser/example/aes_simple.spef";
  //   std::string spef_file_str =
  //       "/home/immelon/projects/scripts_test_ipd/nangate45_example.spef";
  //   std::string spef_file_str =
  //       "/home/immelon/projects/scripts_test_ipd/skywater_aes_cipher_top.spef";

  ista::SpefParser spef_parser;

  bool result = spef_parser.read(spef_file_str);

  std::cout << result << std::endl;
}