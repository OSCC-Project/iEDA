#include <iostream>
#include <cstring>

#include "spef_parser.hh"

#include "spef_parser/src/spef_parser/mod.rs.h"
#include "spef_parser/src/spef_parser/spef_data.rs.h"

namespace ista {
namespace spef {

Parser::Parser() {};

SpefFile Parser::read(rust::String path) {
  return parse_spef_file(path);
};
// There are four parts: header, name map, ports, nets.
// class SpefParserClient::spef_parser_data
// {
  // SpefParserData() {
  //   standard = nullptr;
  //   design_name = nullptr;
  //   date = nullptr;
  //   vendor = nullptr;
  //   program = nullptr;
  //   version = nullptr;
  //   design_flow = nullptr;
  //   divider = nullptr;
  //   delimiter = nullptr;
  //   bus_delimiter= nullptr;
  //   time_unit= nullptr;
  //   capacitance_unit= nullptr;
  //   resistance_unit= nullptr;
  //   inductance_unit= nullptr;
  // }

  // friend SpefParserClient;

  // Header info.
  // std::unordered_map<std::string, std::string> header;
  // std::string standard;
  // SpefHeaderEntry* design_name;
  // SpefHeaderEntry* date;
  // SpefHeaderEntry* vendor;
  // SpefHeaderEntry* program;
  // SpefHeaderEntry* version;
  // SpefHeaderEntry* design_flow;
  // SpefHeaderEntry* divider;
  // SpefHeaderEntry* delimiter;
  // SpefHeaderEntry* bus_delimiter;
  // SpefHeaderEntry* time_unit;
  // SpefHeaderEntry* capacitance_unit;
  // SpefHeaderEntry* resistance_unit;
  // SpefHeaderEntry* inductance_unit;

  // // namemap、ports and nets
  // std::unordered_map<size_t, SpefStringValue> name_map;
  // std::vector<SpefPortEntry> ports;
  // std::vector<SpefNet> nets;
// };

// SpefParserClient::SpefParserClient() : SpefParserData(new class SpefParserClient::SpefParserData) {}

// std::unique_ptr<SpefParserClient> new_spefparser_client() {
//     return std::make_unique<SpefParserClient>();
// }

// Consume a header_entry and add header key-value to spef_parser_data
// void SpefParserClient::consume_header_entry(SpefHeaderEntry &header_entry) {
//   rust::Str header_key = header_entry.get_header_key();
//   rust::Str header_value = header_entry.get_header_value();
//   spef_parser_data->header.insert(header_key, header_value);
// }



}  // namespace spef
}  // namespace ista