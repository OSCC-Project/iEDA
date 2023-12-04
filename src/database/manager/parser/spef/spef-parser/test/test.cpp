#include <cstring>
#include <iostream>

#include "SpefParserRustC.hh"

int main() {
  //   std::string spef_file_str =
  //   "/home/immelon/projects/iPD/src/database/manager/parser/spef/spef-parser/"
  //   "aes.spef";
  std::string spef_file_str =
      "/home/taosimin/skywater130/spef/aes_cipher_top.spef";
  //   std::string spef_file_str =
  //       "/home/immelon/projects/scripts_test_ipd/nangate45_example.spef";
  //   std::string spef_file_str =
  //       "/home/immelon/projects/scripts_test_ipd/skywater_aes_cipher_top.spef";

  ista::SpefRustReader spef_parser;

  bool result = spef_parser.read(spef_file_str);

  // std::cout << "expand name start" << std::endl;
  // spef_parser.expandName();
  // std::cout << "expand name finish" << std::endl;

  // auto* spef_file = spef_parser.get_spef_file();

  // void* spef_net;
  // FOREACH_VEC_ELEM(&(spef_file->_nets), void, spef_net) {
  //   auto* rust_spef_net =
  //       static_cast<RustSpefNet*>(rust_convert_spef_net(spef_net));

  //   std::cout << rust_spef_net->_name << std::endl;
  //   std::cout << rust_spef_net->_lcap << std::endl;

  //   void* spef_net_conn;
  //   FOREACH_VEC_ELEM(&(rust_spef_net->_conns), void, spef_net_conn) {
  //     auto* rust_spef_conn = static_cast<RustSpefConnEntry*>(
  //         rust_convert_spef_conn(spef_net_conn));
  //     std::cout << rust_spef_conn->_name << std::endl;
  //     std::cout << rust_spef_conn->_load << std::endl;
  //     std::cout << rust_spef_conn->_driving_cell << std::endl;
  //     std::cout << rust_spef_conn->_coordinate._x << " "
  //               << rust_spef_conn->_coordinate._y << std::endl;
  //   }

  //   void* spef_net_cap;
  //   FOREACH_VEC_ELEM(&(rust_spef_net->_caps), void, spef_net_cap) {
  //     auto* rust_spef_cap = static_cast<RustSpefResCap*>(
  //         rust_convert_spef_net_cap_res(spef_net_cap));
  //     std::cout << rust_spef_cap->_node1 << std::endl;
  //     std::cout << rust_spef_cap->_node2 << std::endl;
  //     std::cout << rust_spef_cap->_res_or_cap << std::endl;
  //   }

  //   void* spef_net_res;
  //   FOREACH_VEC_ELEM(&(rust_spef_net->_ress), void, spef_net_res) {
  //     auto* rust_spef_cap = static_cast<RustSpefResCap*>(
  //         rust_convert_spef_net_cap_res(spef_net_res));
  //     std::cout << rust_spef_cap->_node1 << std::endl;
  //     std::cout << rust_spef_cap->_node2 << std::endl;
  //     std::cout << rust_spef_cap->_res_or_cap << std::endl;
  //   }
  // }

  // std::cout << spef_parser.getSpefCapUnit() << std::endl;
  // std::cout << spef_parser.getSpefResUnit() << std::endl;

  return result;
}