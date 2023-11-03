#include <cstring>
#include <iostream>
#include "spef_parser.hh"

int main() {
  //   std::string spef_file_str =
  //   "/home/immelon/projects/iPD/src/database/manager/parser/spef/spef-parser/"
  //   "aes.spef";
  std::string spef_file_str =
      "/home/immelon/projects/iPD/src/database/manager/parser/spef/spef-parser/"
      "aes_simple.spef";
  //   std::string spef_file_str =
  //       "/home/immelon/projects/scripts_test_ipd/nangate45_example.spef";
  //   std::string spef_file_str =
  //       "/home/immelon/projects/scripts_test_ipd/skywater_aes_cipher_top.spef";

  ista::spef::Parser spef_parser;

  bool result = spef_parser.read(spef_file_str);

  std::cout << result << std::endl;

}