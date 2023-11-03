#include "spef_parser.hh"

#include <cstring>
#include <iostream>

#include "spef_parser/src/spef_parser/mod.rs.h"
#include "spef_parser/src/spef_parser/spef_data.rs.h"

namespace ista {
namespace spef {

bool Parser::read(rust::String path) {
  try {
    auto result = parse_spef_file(path);

    if (result.name == path) {
      file_data_rust = result;
      return true;
    } else {
      std::cerr << "Error when transferring rust data to cpp" << std::endl;
      return false;
    }
  } catch (const rust::Error& e) {
    std::cerr << "Caught rust error: " << e.what() << std::endl;
    return false;
  }
};

/// @brief Generate unordered_map in cpp from rust Vec<NameMapItem>
/// @param name_map_vec
void Parser::generate_namemap(rust::Vec<NameMapItem> name_map_vec) {
  for (auto item : name_map_vec) {
    name_map.insert(std::make_pair(item.index, item.name));
  }
};

/// @brief Generate and set header info from HeaderItem
/// @param header_vec
void Parser::generate_header(rust::Vec<HeaderItem> header_vec) {
  for (auto item : header_vec) {
    header.insert(std::make_pair(item.key, item.value));
  }
}

/// @brief Convert NetItem to Net
/// @param nets_rust
void Parser::process_nets(rust::Vec<NetItem> nets_rust) {
  for (auto net_rust : nets_rust) {
    std::string name = std::string(net_rust.name);
    float lcap = net_rust.lcap;

    // Extract data in Vec<ConnItem> into a Connection.
    std::vector<Connection> conns;
    for (auto conn_rust : net_rust.conns) {
      Connection conn = Connection();

      // Must-have memebers in a connection
      conn.name = std::string(conn_rust.pin_name);
      conn.type = conn_rust.conn_type;
      conn.direction = conn_rust.conn_direction;
      conn.driving_cell = std::string(conn_rust.driving_cell);

      // Optional memebers in a connection, wrap them in std::optional<>
      // using if-else to compare with default value
      std::optional<std::pair<double, double>> coordinate;
      if ((-1.0, -1.0) !=
          (conn_rust.coordinates[0], conn_rust.coordinates[1])) {
        coordinate = {conn_rust.coordinates[0], conn_rust.coordinates[1]};
      }
      conn.coordinate = coordinate;

      std::optional<double> load;
      if (-1.0 != conn_rust.load) {
        load = conn_rust.load;
      }
      conn.load = load;

      std::optional<std::pair<float, float>> ll_coordinate;
      if ((-1.0, -1.0) !=
          (conn_rust.ll_coordinate[0], conn_rust.ll_coordinate[1])) {
        coordinate = {conn_rust.ll_coordinate[0], conn_rust.ll_coordinate[1]};
      }
      conn.ll_coordinate = ll_coordinate;

      std::optional<std::pair<float, float>> ur_coordinate;
      if ((-1.0, -1.0) !=
          (conn_rust.ur_coordinate[0], conn_rust.ur_coordinate[1])) {
        ur_coordinate = {conn_rust.ur_coordinate[0],
                         conn_rust.ur_coordinate[1]};
      }
      conn.ur_coordinate = ur_coordinate;

      std::optional<int> layer;
      if (0 != conn_rust.layer) {
        layer = conn_rust.layer;
      }
      conn.layer = layer;

      conns.push_back(conn);
    }

    std::vector<std::tuple<std::string, std::string, float>> caps;
    for (auto cap_rust : net_rust.caps) {
      std::tuple<std::string, std::string, double> cap =
          std::make_tuple(std::string(cap_rust.pin_port[0]),
                          std::string(cap_rust.pin_port[1]), cap_rust.cap_val);
      caps.push_back(cap);
    }

    std::vector<std::tuple<std::string, std::string, float>> ress;
    for (auto res_rust : net_rust.ress) {
      std::tuple<std::string, std::string, double> res =
          std::make_tuple(std::string(res_rust.pin_port[0]),
                          std::string(res_rust.pin_port[1]), res_rust.res);
      ress.push_back(res);
    }

    Net current_net = Net(name, lcap);
    current_net.connections = conns;
    current_net.caps = caps;
    current_net.ress = ress;

    nets.push_back(current_net);
  }
}

/// @brief Convert PortItem to Port
/// @param ports_rust 
void Parser::process_ports(rust::Vec<PortItem> ports_rust) {
  for (auto port : ports_rust) {
    
  }
}
}  // namespace spef
}  // namespace ista