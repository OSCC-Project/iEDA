// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file staClockTree.cc
 * @author longshy (longshy@pcl.ac.cn)
 * @brief The implemention of clock tree for GUI show.
 * @version 0.1
 * @date 2023-03-27
 */
#include "StaClockTree.hh"

#include <cstring>
#include <fstream>
#include <regex>

#include "Config.hh"
#include "StaClock.hh"
#include "log/Log.hh"
#include "string/Str.hh"
#include "json/json.hpp"

namespace ista {

ModeTransAT::ModeTransAT(const char* from_name, const char* to_name,
                         double max_rise_to_arrive_time,
                         double max_fall_to_arrive_time,
                         double min_rise_to_arrive_time,
                         double min_fall_to_arrive_time)
    : _from_name(from_name),
      _to_name(to_name),
      _max_rise_to_arrive_time(max_rise_to_arrive_time),
      _max_fall_to_arrive_time(max_fall_to_arrive_time),
      _min_rise_to_arrive_time(min_rise_to_arrive_time),
      _min_fall_to_arrive_time(min_fall_to_arrive_time) {}

StaClockTreeNode::StaClockTreeNode(std::string cell_type, std::string inst_name)
    : _cell_type(cell_type), _inst_name(inst_name) {}

/**
 * @brief get the input_pin_name/max_rise_AT pairs of the node.
 *
 * @return std::vector<StaClockTreeNode::Pin2AT>&
 */
std::vector<StaClockTreeNode::Pin2AT> StaClockTreeNode::getInputPinMaxRiseAT() {
  std::vector<StaClockTreeNode::Pin2AT> input_pin_2_ATS;
  auto& fanin_arcs = get_fanin_arcs();

  // for root node, fanin arc is empt, we use default at.
  if (fanin_arcs.empty()) {
    Pin2AT default_at{get_inst_name_str(), 0.0};
    input_pin_2_ATS.emplace_back(std::move(default_at));
    return input_pin_2_ATS;
  }

  for (auto& fanin_arc : fanin_arcs) {
    auto net_arrive_time = fanin_arc->get_net_arrive_time();
    std::string input_pin_name = net_arrive_time.get_to_name();
    double max_rise_AT = net_arrive_time.get_max_rise_to_arrive_time();
    input_pin_2_ATS.emplace_back(std::make_pair(input_pin_name, max_rise_AT));
  }

  return input_pin_2_ATS;
}

/**
 * @brief get the output_pin_name/max_rise_AT pair of the node.
 *
 * @return StaClockTreeNode::Pin2AT&
 */
std::vector<StaClockTreeNode::Pin2AT>
StaClockTreeNode::getOutputPinMaxRiseAT() {
  std::vector<StaClockTreeNode::Pin2AT> output_pin_2_ATS;

  auto& inst_arrive_times = get_inst_arrive_times();
  for (auto& inst_arrive_time : inst_arrive_times) {
    std::string output_pin_name = inst_arrive_time.get_to_name();
    double max_rise_AT = inst_arrive_time.get_max_rise_to_arrive_time();
    output_pin_2_ATS.emplace_back(std::make_pair(output_pin_name, max_rise_AT));
  }

  return output_pin_2_ATS;
}

StaClockTreeArc::StaClockTreeArc(StaClockTreeNode* parent_node,
                                 StaClockTreeNode* child_node)
    : _parent_node(parent_node), _child_node(child_node) {
  parent_node->addChildNode(child_node);
}

StaClockTreeArc::StaClockTreeArc(const char* net_name,
                                 StaClockTreeNode* parent_node,
                                 StaClockTreeNode* child_node)
    : _net_name(net_name), _parent_node(parent_node), _child_node(child_node) {}

StaClockTree::StaClockTree(StaClock* clock, StaClockTreeNode* root_node)
    : _clock(clock), _root_node(root_node) {}

/**
 * @brief Get the clock tree node accord the inst name.
 *
 * @param inst_name
 * @return StaClockTreeNode*
 */
StaClockTreeNode* StaClockTree::findNode(const char* inst_name) {
  if (strcmp(_root_node->get_inst_name(), inst_name) == 0) {
    return _root_node.get();
  } else {
    for (auto& child_node : _child_nodes) {
      if (strcmp(child_node->get_inst_name(), inst_name) == 0) {
        return child_node.get();
      }
    }
  }

  return nullptr;
}

/**
 * @brief Get the clock tree child node accord the inst name.
 *
 * @param inst_name
 * @return StaClockTreeNode*
 */
StaClockTreeNode* StaClockTree::findChildNode(const char* inst_name) {
  for (auto& child_node : _child_nodes) {
    if (strcmp(child_node->get_inst_name(), inst_name) == 0) {
      return child_node.get();
    }
  }
  return nullptr;
}

/**
 * @brief get the count of child arc with the specified parent node.
 *
 * @param parent_node_name
 * @return StaClockTreeArc*
 */
int StaClockTree::getChildArcCnt(const char* parent_node) {
  std::vector<StaClockTreeArc*> child_arcs;
  for (auto& child_arc : _child_arcs) {
    auto* arc_parent_node = child_arc->get_parent_node();
    if (strcmp(arc_parent_node->get_inst_name(), parent_node) == 0) {
      child_arcs.emplace_back(child_arc.get());
    }
  }
  return child_arcs.size();
}

/**
 * @brief get the all child nodes of the specified parent node.
 *
 * @param parent_node
 * @return std::vector<StaClockTreeNode*>
 */
std::vector<StaClockTreeNode*> StaClockTree::getChildNodes(
    const char* parent_node) {
  std::vector<StaClockTreeNode*> child_nodes;
  for (auto& child_arc : _child_arcs) {
    auto* arc_parent_node = child_arc->get_parent_node();
    auto* arc_child_node = child_arc->get_child_node();
    if (strcmp(arc_parent_node->get_inst_name(), parent_node) == 0) {
      child_nodes.emplace_back(arc_child_node);
    }
  }
  return child_nodes;
}

/**
 * @brief get the node count of every level in the clock tree.
 *
 * @return std::vector<int>
 */
std::vector<int> StaClockTree::getLevelNodeCnt() {
  std::vector<int> child_node_cnt{1};
  auto root_node = _root_node->get_inst_name();
  int child_arc_cnt = getChildArcCnt(root_node);
  child_node_cnt.emplace_back(child_arc_cnt);
  std::vector<StaClockTreeNode*> child_nodes = getChildNodes(root_node);
  getChildNodeCnt(child_nodes, child_node_cnt);
  return child_node_cnt;
}

/**
 * @brief get the child node count of the node.
 *
 * @param child_nodes
 * @param child_node_cnt
 */
void StaClockTree::getChildNodeCnt(std::vector<StaClockTreeNode*> child_nodes,
                                   std::vector<int>& child_node_cnt) {
  if (child_nodes.size() == 0) {
    return;
  }

  int count = 0;
  std::vector<StaClockTreeNode*> next_child_nodes;
  for (auto& child_node : child_nodes) {
    auto child_node1 = child_node->get_inst_name();
    int child_arc1_cnt = getChildArcCnt(child_node1);
    count += child_arc1_cnt;

    std::vector<StaClockTreeNode*> next_child_nodes1 =
        getChildNodes(child_node1);
    for (auto& next_child_node1 : next_child_nodes1) {
      next_child_nodes.emplace_back(next_child_node1);
    }
  }
  child_node_cnt.emplace_back(count);
  getChildNodeCnt(next_child_nodes, child_node_cnt);
}

/**
 * @brief Print the clock tree(a node represents a inst) to graphviz dot file
 * format.
 *
 */
void StaClockTree::printInstGraphViz(const char* file_path,
                                     bool show_port_suffix) {
  auto replace_str = [](const std::string& str, const std::string& old_str,
                        const std::string& new_str) {
    std::regex re(old_str);
    return std::regex_replace(str, re, new_str);
  };

  auto remove_port_suffix = [](const std::string& name) {
    std::regex suffix_re(R"(_[AX]$)");
    return std::regex_replace(name, suffix_re, "");
  };

  LOG_INFO << "dump graph dotviz start";

  std::ofstream dot_file;
  dot_file.open(file_path);
  dot_file << "digraph clocktree {\n";

  for (auto& child_arc : _child_arcs) {
    auto* parent_node = child_arc->get_parent_node();
    auto* child_node = child_arc->get_child_node();

    auto net_arrive_time = child_arc->get_net_arrive_time();
    auto from_name = net_arrive_time.get_from_name();
    from_name = replace_str(from_name, R"(\/)", "_");
    from_name = replace_str(from_name, R"(\:)", "_");
    from_name = replace_str(from_name, R"(\.)", "__");
    auto to_name = net_arrive_time.get_to_name();
    to_name = replace_str(to_name, R"(\/)", "_");
    to_name = replace_str(to_name, R"(\:)", "_");
    to_name = replace_str(to_name, R"(\.)", "__");

    if (!show_port_suffix) {
      from_name = remove_port_suffix(from_name);
      to_name = remove_port_suffix(to_name);
    }

    dot_file << Str::printf("%s[label=\"%s fanout %d\" ]\n", from_name.c_str(),
                            from_name.c_str(),
                            parent_node->get_child_nodes().size());
    dot_file << Str::printf("%s", from_name.c_str()) << " -> "
             << Str::printf("%s", to_name.c_str()) << "\n";

    dot_file << Str::printf("%s[label=\"%s delay %f fanout %d\" ]\n",
                            to_name.c_str(), to_name.c_str(),
                            net_arrive_time.get_max_rise_to_arrive_time(),
                            child_node->get_inst_arrive_times().size());
  }

  dot_file << "}\n";

  dot_file.close();

  LOG_INFO << "dump graph dotviz end";
}

void StaClockTree::printInstJson(const char* file_path, bool show_port_suffix) {
  auto replace_str = [](const std::string& str, const std::string& old_str,
                        const std::string& new_str) {
    std::regex re(old_str);
    return std::regex_replace(str, re, new_str);
  };

  auto remove_port_suffix = [](const std::string& name) {
    std::regex suffix_re(R"(_[AX]$)");
    return std::regex_replace(name, suffix_re, "");
  };

  LOG_INFO << "dump graph json start";

  nlohmann::json graph = nlohmann::json::array();

  for (auto& child_arc : _child_arcs) {
    nlohmann::json node;
    auto* parent_node = child_arc->get_parent_node();
    auto* child_node = child_arc->get_child_node();

    auto net_arrive_time = child_arc->get_net_arrive_time();
    auto from_name = net_arrive_time.get_from_name();
    from_name = replace_str(from_name, R"(\/)", "_");
    from_name = replace_str(from_name, R"(\:)", "_");
    from_name = replace_str(from_name, R"(\.)", "__");
    auto to_name = net_arrive_time.get_to_name();
    to_name = replace_str(to_name, R"(\/)", "_");
    to_name = replace_str(to_name, R"(\:)", "_");
    to_name = replace_str(to_name, R"(\.)", "__");

    if (!show_port_suffix) {
      from_name = remove_port_suffix(from_name);
      to_name = remove_port_suffix(to_name);
    }

    node["name"] = from_name;
    node["to"] = to_name;
    node["fanout"] = parent_node->get_child_nodes().size();
    node["delay"] = net_arrive_time.get_max_rise_to_arrive_time();

    graph.push_back(node);

    if (child_node->get_inst_arrive_times().empty()) {
      graph.push_back({{"name", to_name}, {"fanout", 0}});
    }
  }

  std::ofstream json_file(file_path);
  json_file << graph.dump(4);
  json_file.close();
  LOG_INFO << "dump graph json end";
}

}  // namespace ista