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
 * @file StaCppr.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of cppr(common path pessimism removal).
 * @version 0.1
 * @date 2021-03-24
 */

#include "StaCppr.hh"

namespace ista {

StaCppr::StaCppr(StaClockData* launch_data, StaClockData* capture_data)
    : _launch_data(launch_data), _capture_data(capture_data) {}

/**
 * @brief Get the least common ancestor for clock end1 and clock end2.
 *
 * @param root The clock root node.
 * @param clock_end1 The launch or capture clock endpoint.
 * @param clock_end2 The launch or capture clock endpoint.
 * @return StaVertex* Return the common vertex point.
 */
StaVertex* StaCppr::getLCA(StaVertex* root, StaVertex* clock_end1,
                           StaVertex* clock_end2) {
  if (!root) {
    return nullptr;
  }

  if (root == clock_end1 || root == clock_end2) {
    return root;
  }

  StaVertex* the_node1 = nullptr;  // one clock end parent node.
  StaVertex* the_node2 = nullptr;  // the other clock end parent node.
  auto src_arcs = root->get_src_arcs();
  for (auto* src_arc : src_arcs) {
    auto* snk_node = src_arc->get_snk();
    auto* the_node = getLCA(snk_node, clock_end1, clock_end2);
    if (the_node) {
      if (!the_node1) {
        the_node1 = the_node;
      } else if (!the_node2) {
        the_node2 = the_node;
      } else {
        break;
      }
    }
  }

  if (!the_node1) {
    return the_node2;
  }

  if (!the_node2) {
    return the_node1;
  }
  return root;
}

/**
 * @brief The cppr search of the same clock, which split in the common point
 * to launch and capture clock.
 *
 * @param the_clock The same clock that is launch clock and capture clock.
 * @return unsigned 1 if success, 0 else fail.
 */
unsigned StaCppr::operator()(StaClock* the_clock) {
  auto launch_path_data_stack = _launch_data->getPathData();
  auto capture_path_data_stack = _capture_data->getPathData();

  auto get_path_vertexes = [](auto path_data_stack) -> std::vector<StaVertex*> {
    std::vector<StaVertex*> path_vertexes;
    while (!path_data_stack.empty()) {
      auto* path_data = path_data_stack.top();
      path_vertexes.emplace_back(path_data->get_own_vertex());
      path_data_stack.pop();
    }
    return path_vertexes;
  };

  auto launch_path_vertexes = get_path_vertexes(launch_path_data_stack);
  auto capture_path_vertexes = get_path_vertexes(capture_path_data_stack);

  StaVertex* common_vertex = nullptr;
  for (auto* launch_vertex : launch_path_vertexes) {
    if (std::none_of(capture_path_vertexes.begin(), capture_path_vertexes.end(),
                     [launch_vertex](auto* capture_vertex) {
                       return launch_vertex == capture_vertex;
                     })) {
      break;
    }
    common_vertex = launch_vertex;
  }

  if (!common_vertex) {
    return 1;
  }

  auto find_common_data = [](StaVertex* common_point,
                             StaClockData* clock_data) -> StaClockData* {
    while (clock_data && (clock_data->get_own_vertex() != common_point)) {
      clock_data = dynamic_cast<StaClockData*>(clock_data->get_bwd());
    }

    return clock_data;
  };

  auto* launch_common_data = find_common_data(common_vertex, _launch_data);
  auto* capture_common_data = find_common_data(common_vertex, _capture_data);

  _common_point = common_vertex;

  _cppr = std::abs(launch_common_data->get_arrive_time() -
                   capture_common_data->get_arrive_time());

  return 1;
}

}  // namespace ista
