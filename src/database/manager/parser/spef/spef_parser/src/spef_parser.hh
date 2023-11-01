#pragma once
#include <cstring>
#include <experimental/filesystem>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rust/cxx.h"

namespace ista {
namespace spef {

// Shared or opaque types and structs from rust mod ffi.
struct SpefExchange;
struct SpefFloatValue;
struct SpefStringValue;
struct SpefHeaderEntry;
struct SpefNameMapEntry;
struct SpefPortEntry;
struct SpefConnEntry;
struct SpefNet;

// SpefParserData: the data in a SPEF.
// SpefParserClient: apis to acess SpefParserData.
class SpefParserClient {
  //   struct Error
  //   {
  //     std::string line;
  //     size_t line_number;
  //     size_t byte_in_line;
  //   };
public:
  SpefParserClient();
  std::string get_header_standard() const;
  void consume_header_entry(SpefHeaderEntry& header_entry);
  //   std::optional<Error> error;

  //   std::string dump() const;
  //   std::string dump_compact() const;

  //   void dump(std::ostream&) const;
  //   void dump_compact(std::ostream&) const;
  //   void clear();
  // void expand_name(unsigned num_threads);
  // void expand_name(Net&);
  // void expand_name(Port&);
  // void scale_capacitance(float);
  // void scale_resistance(float);
  // bool read(const std::string);
  // auto& getNets() { return nets; }

  // SpefNet* findSpefNet(const std::string& net_name)
  // {
  //   for (auto& net : nets) {
  //     if (net.name == net_name) {
  //       return &net;
  //     }
  //   }
  //   return nullptr;
  // }
  // std::string get_standard()


 private:
  class spef_parser_data;
  std::shared_ptr<spef_parser_data> spef_parser_data;
};

std::unique_ptr<SpefParserClient> new_spefparser_client();
}  // namespace spef
}  // namespace ista