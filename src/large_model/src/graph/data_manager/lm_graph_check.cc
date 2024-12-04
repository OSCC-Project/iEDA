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
/**
 * @file lm_layout_check.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-11-15
 * @brief Check layout data (connectivity of nets, wires, etc.), and provide interfaces for graph construction.
 */
#include "lm_graph_check.hh"

#include <boost/graph/connected_components.hpp>
#include <boost/graph/graphviz.hpp>
#include <fstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "log/Log.hh"
namespace ilm {

bool GraphCheckerBase::isConnectivity(const Graph& graph) const
{
  std::vector<int> component(boost::num_vertices(graph));
  int num_components = boost::connected_components(graph, &component[0]);

  // If there's only one connected component, the net is fully connected
  return (num_components == 1);
}

void GraphCheckerBase::writeToDot(const Graph& graph, const std::string& path) const
{
  std::ofstream file(path);
  boost::write_graphviz(file, graph, [&](std::ostream& out, const Graph::vertex_descriptor& v) {
    out << "[x=" << std::to_string(graph[v].x) << ", y=" << std::to_string(graph[v].y) << ", layer_id=" << std::to_string(graph[v].layer_id)
        << ", pin_id=" << std::to_string(graph[v].pin_id) << "]";
  });
}

void GraphCheckerBase::writeToPy(LmNet& net, const std::string& path) const
{
  // Write the wire line to a Python file for plotting in 3D space (x, y, layer_id) using Plotly
  std::ofstream file(path);
  file << "import plotly.graph_objects as go\n";
  file << "\n";
  file << "# Create a Plotly Figure\n";
  file << "fig = go.Figure()\n";
  file << "\n";

  // Add data for each wire and path
  auto& wires = net.get_wires();
  for (auto& wire : wires) {
    auto& paths = wire.get_paths();
    for (auto& path : paths) {
      auto* start = path.first;
      auto* end = path.second;

      // Add a trace for each path
      file << "fig.add_trace(go.Scatter3d(\n";
      file << "    x=[" << start->get_x() << ", " << end->get_x() << "],\n";
      file << "    y=[" << start->get_y() << ", " << end->get_y() << "],\n";
      file << "    z=[" << start->get_layer_id() << ", " << end->get_layer_id() << "],\n";
      file << "    mode='lines',\n";
      file << "    line=dict(color='rgb(" << (50 + end->get_layer_id() * 20) % 256 << "," << (100 + end->get_layer_id() * 30) % 256 << ","
           << (150 + end->get_layer_id() * 40) % 256 << ")', width=4),\n";
      file << "))\n";
    }
  }

  // Set up the layout with axis labels and title
  file << "\n";
  file << "fig.update_layout(\n";
  file << "    scene=dict(\n";
  file << "        xaxis_title='X',\n";
  file << "        yaxis_title='Y',\n";
  file << "        zaxis_title='Layer ID'\n";
  file << "    ),\n";
  file << "    title='Net " << net.get_net_id() << "'\n";
  file << ")\n";
  file << "\n";

  // Show the plot
  file << "fig.show()\n";
}

void GraphCheckerBase::writeToPy(const Graph& graph, LmNet& net, const std::string& path, const bool& mark_break,
                                 const bool& mark_pin_id) const
{
  // get all
  auto& wires = net.get_wires();
  // build a map for get wire by (x1,y1,layer_id1) and (x2,y2,layer_id2)
  struct WireKey
  {
    int x1;
    int y1;
    int layer_id1;
    int pin_id1;
    int x2;
    int y2;
    int layer_id2;
    int pin_id2;
    bool operator==(const WireKey& other) const
    {
      return x1 == other.x1 && y1 == other.y1 && layer_id1 == other.layer_id1 && pin_id1 == other.pin_id1 && x2 == other.x2
             && y2 == other.y2 && layer_id2 == other.layer_id2 && pin_id2 == other.pin_id2;
    }
  };
  struct WireKeyHash
  {
    std::size_t operator()(const WireKey& key) const
    {
      return (std::hash<int>()(key.x1) ^ std::hash<int>()(key.y1) ^ std::hash<int>()(key.layer_id1) ^ std::hash<int>()(key.pin_id1)
              ^ std::hash<int>()(key.x2) ^ std::hash<int>()(key.y2) ^ std::hash<int>()(key.layer_id2) ^ std::hash<int>()(key.pin_id2));
    }
  };
  std::unordered_map<WireKey, LmNetWire, WireKeyHash> wire_map;
  for (auto& wire : wires) {
    auto& paths = wire.get_paths();
    for (auto& path : paths) {
      auto* start = path.first;
      auto* end = path.second;
      WireKey key = {start->get_x(), start->get_y(), start->get_layer_id(), start->get_node_data()->get_pin_id(),
                     end->get_x(),   end->get_y(),   end->get_layer_id(),   end->get_node_data()->get_pin_id()};
      wire_map[key] = wire;
      WireKey key_reverse = {end->get_x(),   end->get_y(),   end->get_layer_id(),   end->get_node_data()->get_pin_id(),
                             start->get_x(), start->get_y(), start->get_layer_id(), start->get_node_data()->get_pin_id()};
      wire_map[key_reverse] = wire;
    }
  }

  // Write the wire line to a Python file for plotting in 3D space (x, y, layer_id) using Plotly
  std::ofstream file(path);
  file << "import plotly.graph_objects as go\n";
  file << "import matplotlib.cm as cm\n";
  file << "\n";
  file << "# Create a Plotly Figure\n";
  file << "fig = go.Figure()\n";
  file << "\n";

  // for each component, plot the subgraph edges' label (x,y,layer_id), use the same color for the same component
  std::vector<size_t> component(boost::num_vertices(graph));
  size_t num_components = boost::connected_components(graph, &component[0]);
  file << "colors = [f'rgb{cm.tab10(i)[:3]}' for i in range(" << num_components << ")]\n";
  for (size_t i = 0; i < num_components; ++i) {
    file << "# Component " << i << "\n";
    // for each wire, plot the wire with the same color
    size_t wire_idx = 0;
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
      auto u = boost::source(e, graph);
      auto v = boost::target(e, graph);
      if (component[u] == i && component[v] == i) {
        // file << "fig.add_trace(go.Scatter3d(\n";
        // file << "    x=[" << graph[u].x << ", " << graph[v].x << "],\n";
        // file << "    y=[" << graph[u].y << ", " << graph[v].y << "],\n";
        // file << "    z=[" << graph[u].layer_id << ", " << graph[v].layer_id << "],\n";
        // file << "    mode='lines',\n";
        // file << "    line=dict(color=colors[" << i << "], width=4),\n";
        // file << "    name='Component " << i << ", Wire " << wire_idx++ << "'\n";
        // file << "))\n";
        auto wire = wire_map[{graph[u].x, graph[u].y, graph[u].layer_id, graph[u].pin_id, graph[v].x, graph[v].y, graph[v].layer_id,
                              graph[v].pin_id}];
        size_t path_idx = 0;
        for (auto& path : wire.get_paths()) {
          auto* start = path.first;
          auto* end = path.second;
          file << "fig.add_trace(go.Scatter3d(\n";
          file << "    x=[" << start->get_x() << ", " << end->get_x() << "],\n";
          file << "    y=[" << start->get_y() << ", " << end->get_y() << "],\n";
          file << "    z=[" << start->get_layer_id() << ", " << end->get_layer_id() << "],\n";
          file << "    mode='lines',\n";
          file << "    line=dict(color=colors[" << i << "], width=4),\n";
          file << "    name='Component " << i << ", Wire " << wire_idx << ", Path " << path_idx++ << "'\n";
          file << "))\n";
        }
        wire_idx++;
      }
    }
  }

  // plot the nodes which degree is 1, plot with 'x' marker
  if (mark_break) {
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
      if (boost::degree(v, graph) == 1) {
        file << "fig.add_trace(go.Scatter3d(\n";
        file << "    x=[" << graph[v].x << "],\n";
        file << "    y=[" << graph[v].y << "],\n";
        file << "    z=[" << graph[v].layer_id << "],\n";
        file << "    mode='markers',\n";
        file << "    marker=dict(\n";
        file << "        size=3,\n";
        file << "        symbol='x',\n";
        file << "        color='red'\n";
        file << "    )\n";
        file << "))\n";
      }
    }
  }

  // plot the nodes with pin_id, plot with text
  if (mark_pin_id) {
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
      if (graph[v].pin_id != -1) {
        file << "fig.add_trace(go.Scatter3d(\n";
        file << "    x=[" << graph[v].x << "],\n";
        file << "    y=[" << graph[v].y << "],\n";
        file << "    z=[" << graph[v].layer_id << "],\n";
        file << "    mode='text',\n";
        file << "    text=['" << graph[v].pin_id << "'],\n";
        file << "    textposition='top center',\n";
        file << "    textfont=dict(\n";
        file << "        size=12,\n";
        file << "        color='black'\n";
        file << "    )\n";
        file << "))\n";
      }
    }
  }

  // Set up the layout with axis labels and title
  file << "\n";
  file << "fig.update_layout(\n";
  file << "    scene=dict(\n";
  file << "        xaxis_title='X',\n";
  file << "        yaxis_title='Y',\n";
  file << "        zaxis_title='Layer ID'\n";
  file << "    ),\n";
  file << "    title='Net " << net.get_net_id() << "'\n";
  file << ")\n";
  file << "\n";

  // Show the plot
  file << "fig.show()\n";
}

bool LmNetChecker::isLocalConnectivity(LmNet& net) const
{
  auto& wires = net.get_wires();
  // 1. Check wire inner connectivity
  for (auto& wire : wires) {
    if (!isLocalConnectivity(wire)) {
      return false;
    }
  }

  // 2. Convert the net to a graph
  auto graph = convertToGraph(net);

  // 3. Check if the entire graph is connected
  return GraphCheckerBase::isConnectivity(graph);
}

bool LmNetChecker::isLocalConnectivity(LmNetWire& wire) const
{
  std::vector<std::pair<LmNode*, LmNode*>> paths = wire.get_paths();
  auto is_same_loc = [](LmNode* node1, LmNode* node2) { return node1->get_x() == node2->get_x() && node1->get_y() == node2->get_y(); };
  for (size_t i = 0; i < paths.size() - 1; ++i) {
    auto* front = paths[i].second;
    auto* back = paths[i + 1].first;
    // avoid empty pointer
    if (!front || !back) {
      return false;
    }
    if (!is_same_loc(front, back)) {
      return false;
    }
  }
  return true;
}
Graph LmNetChecker::convertToGraph(LmNet& net) const
{
  // 1. Create a graph based on the net
  // Map to assign unique IDs to connection points
  std::unordered_map<std::tuple<int, int, int32_t>, size_t, boost::hash<std::tuple<int, int, int32_t>>> node_to_id;
  size_t next_id = 0;
  auto& wires = net.get_wires();
  // First pass: Assign unique IDs to each connection point
  for (auto& wire : wires) {
    auto& connected_nodes = wire.get_connected_nodes();
    auto* start = connected_nodes.first;
    auto* end = connected_nodes.second;

    // Create keys based on (x, y, layer_id)
    auto start_key = std::make_tuple(start->get_x(), start->get_y(), start->get_layer_id());
    auto end_key = std::make_tuple(end->get_x(), end->get_y(), end->get_layer_id());

    // Assign unique ID if not already assigned
    if (!node_to_id.contains(start_key)) {
      node_to_id[start_key] = next_id++;
    }
    if (!node_to_id.contains(end_key)) {
      node_to_id[end_key] = next_id++;
    }
  }

  // Create a graph with the number of unique connection points as vertices
  Graph graph(next_id);
  std::unordered_map<Edge, LmNetWire, EdgeHash> edge_to_wire;

  // Second pass: Add edges between connection points based on wires
  for (auto& wire : wires) {
    auto& connected_nodes = wire.get_connected_nodes();
    auto* start = connected_nodes.first;
    auto* end = connected_nodes.second;

    auto start_key = std::make_tuple(start->get_x(), start->get_y(), start->get_layer_id());
    auto end_key = std::make_tuple(end->get_x(), end->get_y(), end->get_layer_id());

    auto start_id = node_to_id[start_key];
    auto end_id = node_to_id[end_key];

    auto start_vertex = boost::vertex(start_id, graph);
    graph[start_vertex].x = start->get_x();
    graph[start_vertex].y = start->get_y();
    graph[start_vertex].layer_id = start->get_layer_id();
    graph[start_vertex].pin_id = start->get_node_data()->get_pin_id();

    auto end_vertex = boost::vertex(end_id, graph);
    graph[end_vertex].x = end->get_x();
    graph[end_vertex].y = end->get_y();
    graph[end_vertex].layer_id = end->get_layer_id();
    graph[end_vertex].pin_id = end->get_node_data()->get_pin_id();

    auto [edge, inserted] = boost::add_edge(start_id, end_id, graph);

    edge_to_wire[edge] = wire;
  }

  // 2. Remove redundant components, if have one component contains all pins, remove all other components (and correspondingly wires)
  // Get all pins
  auto& pin_ids = net.get_pin_ids();
  std::sort(pin_ids.begin(), pin_ids.end());

  // Get all components
  std::vector<int> component(boost::num_vertices(graph));
  int num_components = boost::connected_components(graph, &component[0]);

  // Try to find the component containing all pins
  int desired_component = -1;
  for (int component_id = 0; component_id < num_components; ++component_id) {
    std::unordered_set<int> pins_in_component;
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
      if (component[v] == component_id) {
        auto pin_id = graph[v].pin_id;
        if (pin_id != -1) {  // pin node
          pins_in_component.insert(pin_id);
        }
      }
    }
    auto pins_in_component_vec = std::vector<int>(pins_in_component.begin(), pins_in_component.end());
    std::sort(pins_in_component_vec.begin(), pins_in_component_vec.end());

    if (pins_in_component_vec == pin_ids) {
      desired_component = component_id;
      break;
    }
  }
  if (desired_component == -1) {
    // Not fonud
    return graph;
  }

  // Remove all other components
  std::vector<Vertex> vertices_to_remove;
  for (int component_id = 0; component_id < num_components; ++component_id) {
    if (component_id != desired_component) {
      for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        if (component[v] == component_id) {
          vertices_to_remove.push_back(v);
        }
      }
    }
  }

  // Update the component's correspondingly wires
  std::vector<LmNetWire> updated_wires;
  for (auto e : boost::make_iterator_range(boost::edges(graph))) {
    if (component[boost::source(e, graph)] == desired_component) {
      auto wire = edge_to_wire[e];
      updated_wires.push_back(wire);
    }
  }

  std::sort(vertices_to_remove.begin(), vertices_to_remove.end(), [&](const Vertex a, const Vertex b) { return a > b; });
  for (auto v : vertices_to_remove) {
    boost::clear_vertex(v, graph);
    boost::remove_vertex(v, graph);
  }

  wires = updated_wires;
  return graph;
}
bool LmLayoutChecker::checkLayout(std::map<int, LmNet> net_map)
{
  int success_num = 0;
  int total = 0;
  for (auto& [net_id, net] : net_map) {
    if (addNet(net)) {
      success_num++;
    }
    total++;
  }

  LOG_ERROR << "Net connected succuss ratio : " << success_num << " / " << total;
  return isConnectivity();
}
bool LmLayoutChecker::addNet(LmNet& net)
{
  // check local connectivity
  auto checker = LmNetChecker();
  if (!checker.isLocalConnectivity(net)) {
    LOG_ERROR << "Net " << net.get_net_id() << " is not locally connected.";
    // debug
    auto graph = checker.convertToGraph(net);
    // GraphCheckerBase::writeToDot(
    //     graph, "/data/project_share/benchmark/t28/baseline/result/feature/graph_debug_temp/net_" + std::to_string(net.get_net_id()) +
    //     ".dot");
    GraphCheckerBase::writeToPy(
        graph, net,
        "/data/project_share/benchmark/t28/baseline/result/feature/graph_debug/net_" + std::to_string(net.get_net_id()) + ".py");
    return false;
  } else {
    LOG_INFO << "Net " << net.get_net_id() << " is locally connected.";
    // debug
    // auto graph = checker.convertToGraph(net);
    // GraphCheckerBase::writeToDot(
    //     graph, "/data/project_share/benchmark/t28/baseline/result/feature/graph_debug/net_" + std::to_string(net.get_net_id()) + ".dot");
    // GraphCheckerBase::writeToPy(
    //     graph, net,
    //     "/data/project_share/benchmark/t28/baseline/result/feature/graph_debug/net_" + std::to_string(net.get_net_id()) + "_pass.py");
  }
  _nets.push_back(net);
  return true;
}
bool LmLayoutChecker::isConnectivity()
{
  _node_id = 0;
  _node_to_id.clear();
  _graph.clear();

  // 1. convert all nets' nodes to id
  for (auto& net : _nets) {
    auto& wires = net.get_wires();
    for (auto& wire : wires) {
      auto& connected_nodes = wire.get_connected_nodes();
      auto* start = connected_nodes.first;
      auto* end = connected_nodes.second;

      // Create keys based on (x, y, layer_id)
      auto start_key = std::make_tuple(start->get_x(), start->get_y(), start->get_layer_id());
      auto end_key = std::make_tuple(end->get_x(), end->get_y(), end->get_layer_id());

      // Assign unique ID if not already assigned
      if (!_node_to_id.contains(start_key)) {
        _node_to_id[start_key] = _node_id++;
      }
      if (!_node_to_id.contains(end_key)) {
        _node_to_id[end_key] = _node_id++;
      }
    }
  }
  // 2. convert all nets to graph
  _graph = Graph(_node_id);
  for (auto& net : _nets) {
    auto& wires = net.get_wires();
    for (auto& wire : wires) {
      auto& connected_nodes = wire.get_connected_nodes();
      auto* start = connected_nodes.first;
      auto* end = connected_nodes.second;

      auto start_key = std::make_tuple(start->get_x(), start->get_y(), start->get_layer_id());
      auto end_key = std::make_tuple(end->get_x(), end->get_y(), end->get_layer_id());

      auto start_id = _node_to_id[start_key];
      auto end_id = _node_to_id[end_key];

      auto start_vertex = boost::vertex(start_id, _graph);
      _graph[start_vertex].x = start->get_x();
      _graph[start_vertex].y = start->get_y();
      _graph[start_vertex].layer_id = start->get_layer_id();
      _graph[start_vertex].pin_id = start->get_node_data()->get_pin_id();

      auto end_vertex = boost::vertex(end_id, _graph);
      _graph[end_vertex].x = end->get_x();
      _graph[end_vertex].y = end->get_y();
      _graph[end_vertex].layer_id = end->get_layer_id();
      _graph[end_vertex].pin_id = end->get_node_data()->get_pin_id();

      boost::add_edge(start_id, end_id, _graph);
    }
  }
  // 3. check the connectivity of the graph
  if (GraphCheckerBase::isConnectivity(_graph)) {
    LOG_INFO << "The layout is fully connected.";
    return true;
  }
  LOG_ERROR << "The layout is not fully connected.";
  // debug
  return false;

  // sort the components by number of nodes (ascending)
  std::vector<size_t> component(boost::num_vertices(_graph));
  size_t num_components = boost::connected_components(_graph, &component[0]);
  std::unordered_map<int, int> component_size;
  for (size_t i = 0; i < num_components; ++i) {
    component_size[i] = 0;
  }
  for (size_t i = 0; i < component.size(); ++i) {
    component_size[component[i]]++;
  }
  std::vector<std::pair<int, int>> sorted_component;
  for (auto& [key, value] : component_size) {
    sorted_component.push_back(std::make_pair(key, value));
  }
  std::sort(sorted_component.begin(), sorted_component.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.second < b.second; });
  // print the components, save the result to graphviz (dot) file
  std::ofstream ofs("layout.dot");
  ofs << "graph G {\n";
  for (size_t i = 0; i < num_components; ++i) {
    ofs << "subgraph cluster_" << i << " {\n";
    ofs << "label = \"Component " << i << " (" << component_size[i] << " nodes)\";\n";
    for (size_t j = 0; j < component.size(); ++j) {
      if (component[j] == i) {
        ofs << j << ";\n";
      }
    }
    ofs << "}\n";
  }

  return false;
}
}  // namespace ilm