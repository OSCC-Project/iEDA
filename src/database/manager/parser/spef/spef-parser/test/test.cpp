#include <cstring>
#include <iostream>

#include "SpefParserRustC.hh"

int main() {
  //   std::string spef_file_str =
  //   "/home/immelon/projects/iPD/src/database/manager/parser/spef/spef-parser/"
  //   "aes.spef";
  std::string spef_file_str =
      "/home/taosimin/iEDA/src/database/manager/parser/spef/spef-parser/"
      "example/aes_simple.spef";
  //   std::string spef_file_str =
  //       "/home/immelon/projects/scripts_test_ipd/nangate45_example.spef";
  //   std::string spef_file_str =
  //       "/home/immelon/projects/scripts_test_ipd/skywater_aes_cipher_top.spef";

  ista::SpefRustReader spef_parser;

  bool result = spef_parser.read(spef_file_str);

  auto* spef_file = spef_parser.get_spef_file();

  void* spef_net;
  FOREACH_VEC_ELEM(&(spef_file->_nets), void, spef_net) {
    auto* rust_spef_net =
        static_cast<RustSpefNet*>(rust_convert_spef_net(spef_net));

    std::cout << rust_spef_net->_name << std::endl;
    std::cout << rust_spef_net->_lcap << std::endl;
  }

  std::cout << result << std::endl;
}