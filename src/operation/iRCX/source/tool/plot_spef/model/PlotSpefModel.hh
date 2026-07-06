// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace ircx::plot_spef {

struct Node
{
  std::string name;
  int layer = 0;
  int x = 0;
  int y = 0;
  int llx = 0;
  int lly = 0;
  int urx = 0;
  int ury = 0;
  bool has_point = false;
  bool has_box = false;
  bool visible = true;
};

struct Resistor
{
  std::string node1;
  std::string node2;
  double value = 0.0;
  std::size_t index = 0;
  double length = 0.0;
  double width = 0.0;
  int layer = 0;
  int direction = -1;
  int llx = 0;
  int lly = 0;
  int urx = 0;
  int ury = 0;
  bool has_length = false;
  bool has_width = false;
  bool has_layer = false;
  bool has_direction = false;
  bool has_box = false;
  bool visible = true;
};

inline auto resistorLength(const Resistor& resistor) -> double
{
  return resistor.has_length ? resistor.length : 0.0;
}

inline auto resistorWidth(const Resistor& resistor) -> double
{
  return resistor.has_width ? resistor.width : 0.0;
}

inline auto isWireResistor(const Resistor& resistor) -> bool
{
  return resistor.has_layer && resistor.layer >= 1 && resistor.layer <= 10 && resistorLength(resistor) > 1e-12 && resistorWidth(resistor) < 1.0;
}

struct EdgeRef
{
  std::size_t net_index = 0;
  std::size_t resistor_index = 0;
  bool valid = false;
};

inline auto edgeRefKey(const EdgeRef& ref) -> std::string
{
  return std::to_string(ref.net_index) + ":" + std::to_string(ref.resistor_index);
}

inline auto edgeRefTieValue(const EdgeRef& ref) -> std::size_t
{
  return ref.net_index * 1000000U + ref.resistor_index;
}

inline auto nodeEdgeVoteKey(const std::string& node, const EdgeRef& ref) -> std::string
{
  return node + "\n" + edgeRefKey(ref);
}

struct Capacitor
{
  std::string node1;
  std::string node2;
  double value = 0.0;
  EdgeRef edge1;
  EdgeRef edge2;
};

struct Net
{
  std::string name;
  bool visible = true;
  bool context_only = false;
  std::vector<Node> nodes;
  std::unordered_map<std::string, Node*> nodes_by_name;
  std::vector<Resistor> resistors;
  std::vector<Capacitor> coupling_caps;
  std::vector<Capacitor> ground_caps;
};

struct Model
{
  std::string design_name = "plot_spef";
  std::string vendor_name;
  std::string program_name;
  std::string cap_unit;
  std::string res_unit;
  int dbu = 1000;
  std::vector<Net> nets;
  std::unordered_map<std::string, Node*> nodes_by_name;
  std::unordered_map<int, std::string> layer_names;
};

}  // namespace ircx::plot_spef
