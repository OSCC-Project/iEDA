#pragma once
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ista {

extern "C" {
void rust_parser_spef(const char* spef_path);
}

// Port: the port in *PORTS section
struct SpefPort
{
  SpefPort() = default;
  SpefPort(const std::string& s) : name(s) {}
  std::string name;
  // ConnectionDirection direction;  // I, O, B
};

// Connection: the *CONN section in *D_NET
struct SpefConnection
{
  std::string name;
  // ConnectionType type;
  // ConnectionDirection direction;

  std::optional<std::pair<float, float>> coordinate;
  std::optional<float> load;
  std::string driving_cell;
  std::optional<std::pair<float, float>> ll_coordinate;
  std::optional<std::pair<float, float>> ur_coordinate;
  std::optional<int> layer;

  void scale_capacitance(float);
};

// Net: the data in a *D_NET section
//   - Capacitor can be ground (one node) or coupled (two nodes)
struct SpefNet
{
  std::string name;
  float lcap;
  std::vector<SpefConnection> connections;
  std::vector<std::tuple<std::string, std::string, float>> caps;
  std::vector<std::tuple<std::string, std::string, float>> ress;

  SpefNet() = default;
  SpefNet(const std::string& s, const float f) : name{s}, lcap{f} {}

  void scale_capacitance(float);
  void scale_resistance(float);
};

class SpefParser
{
 public:
  std::unordered_map<std::string, std::string> header;
  std::unordered_map<size_t, std::string> name_map;
  std::vector<SpefPort> ports;
  std::vector<SpefNet> nets;

  bool read(std::string file_path);
  void process_rust_data();
  void expand_name(unsigned num_threads);

  //   void generate_namemap(rust::Vec<NameMapItem> name_map_vec);
  //   void generate_header(rust::Vec<HeaderItem> header_vec);

  //   void process_nets(rust::Vec<NetItem> nets_rust);
  //   void process_ports(rust::Vec<PortItem> ports_rust);

  //  private:
  //   SpefFile file_data_rust;
};

}  // namespace ista