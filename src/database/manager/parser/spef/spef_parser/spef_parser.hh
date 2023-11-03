#pragma once
#include <cstring>
// #include <experimental/filesystem>
// #include <memory>
// #include <string_view>
#include <unordered_map>
#include <optional>
// #include <vector>

#include "include/cxx.h"
#include "spef_parser/src/spef_parser/mod.rs.h"
#include "spef_parser/src/spef_parser/spef_data.rs.h"

namespace ista {
namespace spef {

// Shared or opaque types and structs from rust mod ffi.
struct SpefFile;
struct HeaderItem;
struct NameMapItem;
struct PortItem;
struct ConnItem;
struct NetItem;
// enum ConnectionType;
// enum ConnectionDirection;

// Port: the port in *PORTS section
struct Port
{
  Port() = default;
  Port(const std::string& s) : name(s) {}
  std::string name;
  ConnectionDirection direction;  // I, O, B
};

// Connection: the *CONN section in *D_NET
struct Connection
{
  std::string name;
  ConnectionType type;
  ConnectionDirection direction;

  std::optional<std::pair<float, float>> coordinate;
  std::optional<float> load;
  std::string driving_cell;
  std::optional<std::pair<float, float>> ll_coordinate;
  std::optional<std::pair<float, float>> ur_coordinate;
  std::optional<int> layer;

  Connection() = default;

  void scale_capacitance(float);
};

// Net: the data in a *D_NET section
//   - Capacitor can be ground (one node) or coupled (two nodes)
struct Net
{
  std::string name;
  float lcap;
  std::vector<Connection> connections;
  std::vector<std::tuple<std::string, std::string, float>> caps;
  std::vector<std::tuple<std::string, std::string, float>> ress;

  Net() = default;
  Net(const std::string& s, const float f) : name{s}, lcap{f} {}

  void scale_capacitance(float);
  void scale_resistance(float);
};

class Parser {
 public:
  Parser() {};

  std::unordered_map<std::string, std::string> header;
  std::unordered_map<size_t, std::string> name_map;
  std::vector<Port> ports;
  std::vector<Net> nets;

  bool read(rust::String path);
  void process_rust_data();
  void expand_name(unsigned num_threads);

  void generate_namemap(rust::Vec<NameMapItem> name_map_vec);
  void generate_header(rust::Vec<HeaderItem> header_vec);

  void process_nets(rust::Vec<NetItem> nets_rust);
  void process_ports(rust::Vec<PortItem> ports_rust);
 
 private:
  SpefFile file_data_rust;
};

}  // namespace spef
}  // namespace ista