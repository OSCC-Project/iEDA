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
 * @file StaBuildClockTree.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2023-09-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "StaBuildClockTree.hh"

namespace ista {

/**
 * @brief Build the next pin in clock tree.
 *
 */
void StaBuildClockTree::buildNextPin(
    StaClockTree *clock_tree, StaClockTreeNode *parent_node,
    StaVertex *parent_vertex,
    std::map<StaVertex *, std::vector<StaData *>> &vertex_to_datas) {
  for (auto &[fwd_vertex, fwd_datas] : vertex_to_datas) {
    std::map<StaVertex *, std::vector<StaData *>> next_vertex_to_datas;

    double max_rise_AT = 0;
    double max_fall_AT = 0;
    double min_rise_AT = 0;
    double min_fall_AT = 0;
    for (auto *fwd_clock_data : fwd_datas) {
      if ((fwd_clock_data->get_delay_type() == AnalysisMode::kMax) &&
          (fwd_clock_data->get_trans_type() == TransType::kRise)) {
        max_rise_AT =
            (dynamic_cast<StaClockData *>(fwd_clock_data))->get_arrive_time();
        max_rise_AT = FS_TO_NS(max_rise_AT);
      } else if ((fwd_clock_data->get_delay_type() == AnalysisMode::kMax) &&
                 (fwd_clock_data->get_trans_type() == TransType::kFall)) {
        max_fall_AT =
            (dynamic_cast<StaClockData *>(fwd_clock_data))->get_arrive_time();
        max_fall_AT = FS_TO_NS(max_fall_AT);
      } else if ((fwd_clock_data->get_delay_type() == AnalysisMode::kMin) &&
                 (fwd_clock_data->get_trans_type() == TransType::kRise)) {
        min_rise_AT =
            (dynamic_cast<StaClockData *>(fwd_clock_data))->get_arrive_time();
        min_rise_AT = FS_TO_NS(min_rise_AT);
      } else {
        min_fall_AT =
            (dynamic_cast<StaClockData *>(fwd_clock_data))->get_arrive_time();
        min_fall_AT = FS_TO_NS(min_fall_AT);
      }

      for (auto *next_fwd_clock_data : fwd_clock_data->get_fwd_set()) {
        auto *next_fwd_vertex = next_fwd_clock_data->get_own_vertex();
        next_vertex_to_datas[next_fwd_vertex].emplace_back(next_fwd_clock_data);
      }
    }

    std::string from_name = parent_vertex->getName();
    std::string to_name = fwd_vertex->getName();
    ModeTransAT mode_trans_AT(from_name.c_str(), to_name.c_str(), max_rise_AT,
                              max_fall_AT, min_rise_AT, min_fall_AT);

    std::string parent_cell_type = parent_node->get_cell_type();
    // build clock node, annotate delay
    std::string child_cell_type = fwd_vertex->getOwnCellOrPortName();
    std::string child_inst_name = fwd_vertex->getOwnInstanceOrPortName();

    std::string parent_inst_name = parent_vertex->getOwnInstanceOrPortName();
    StaClockTreeNode *child_node =
        clock_tree->findNode(child_inst_name.c_str());
    bool is_new = false;
    if (!child_node) {
      is_new = true;
      child_node = new StaClockTreeNode(child_cell_type, child_inst_name);
      clock_tree->addChildNode(child_node);
    }

    if (parent_node != child_node) {
      auto *child_arc = new StaClockTreeArc(parent_node, child_node);
      child_arc->set_net_arrive_time(mode_trans_AT);
      child_node->addFaninArc(child_arc);
      clock_tree->addChildArc(child_arc);
    } else {
      parent_node->addInstArrvieTime(std::move(mode_trans_AT));
    }

    if ((parent_node != child_node) && (is_new == false)) {
      return;
    }

    if (fwd_vertex->is_clock()) {
      return;
    }

    buildNextPin(clock_tree, child_node, fwd_vertex, next_vertex_to_datas);
  }
}

/**
 * @brief build clock tree for the clock.
 *
 * @param the_clock
 * @return unsigned
 */
unsigned StaBuildClockTree::operator()(StaClock *the_clock) {
  // get_clock_vertexs: usually return one.
  auto &vertexes = the_clock->get_clock_vertexes();

  for (auto *vertex : vertexes) {
    // for each vertex, make one root_node/clock_tree.
    std::string pin_name = vertex->getName();
    std::string cell_type = vertex->getOwnCellOrPortName();
    std::string inst_name = vertex->getOwnInstanceOrPortName();
    auto *root_node = new StaClockTreeNode(cell_type, inst_name);
    auto *clock_tree = new StaClockTree(the_clock, root_node);
    addClockTree(clock_tree);

    StaData *clock_data;
    std::map<StaVertex *, std::vector<StaData *>> next_vertex_to_datas;
    FOREACH_CLOCK_DATA(vertex, clock_data) {
      if ((dynamic_cast<StaClockData *>(clock_data))->get_prop_clock() !=
          clock_tree->get_clock()) {
        continue;
      }

      for (auto *next_fwd_clock_data : clock_data->get_fwd_set()) {
        auto *next_fwd_vertex = next_fwd_clock_data->get_own_vertex();
        next_vertex_to_datas[next_fwd_vertex].emplace_back(next_fwd_clock_data);
      }
    }
    buildNextPin(clock_tree, root_node, vertex, next_vertex_to_datas);
  }

  return 1;
}

}  // namespace ista
