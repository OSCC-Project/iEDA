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
#include "MultiPatterning.h"
namespace idrc {
// std::vector<std::vector<DrcConflictNode*>> MultiPatterning::checkDoublePatterning()
// {
//   std::vector<DrcConflictGraph*> connected_component_list = _connected_component_finder.getAllConnectedComponentInGraph(_conflict_graph);
//   std::vector<std::vector<DrcConflictNode*>> odd_cycle_list = _odd_cycle_finder.findAllOddCycles(connected_component_list);
//   return odd_cycle_list;
// }

std::vector<DrcConflictNode*> MultiPatterning::checkTriplePatterning()
{
  std::vector<DrcConflictGraph*> connected_component_list = _connected_component_finder.getAllConnectedComponentInGraph(_conflict_graph);
  _colorable_checker.set_optional_color_num(3);
  std::vector<DrcConflictNode*> uncolorable_node_list = _colorable_checker.colorable_check(connected_component_list);
  return uncolorable_node_list;
}

std::vector<DrcConflictNode*> MultiPatterning::checkMultiPatterning(int check_colorable_num)
{
  std::vector<DrcConflictGraph*> connected_component_list = _connected_component_finder.getAllConnectedComponentInGraph(_conflict_graph);
  _colorable_checker.set_optional_color_num(check_colorable_num);
  std::cout << "sub graph num ::" << connected_component_list.size() << std::endl;
  std::vector<DrcConflictNode*> uncolorable_node_list = _colorable_checker.colorable_check(connected_component_list);
  repotResult(uncolorable_node_list, check_colorable_num);
  std::cout << "uncolorable node num ::" << uncolorable_node_list.size() << std::endl;
  return uncolorable_node_list;
}

void MultiPatterning::repotResult(std::vector<DrcConflictNode*>& uncolorable_node_list, int optional_color_num)
{
  std::ostringstream oss;
  oss << "MultiPatterning.txt";
  std::string spot_file_name = oss.str();
  oss.str("");
  std::ofstream spot_file(spot_file_name);
  assert(spot_file.is_open());

  spot_file << "**********************************" << std::endl;
  spot_file << "optional color num :: " << optional_color_num << std::endl;
  spot_file << "uncolorable node num :: " << uncolorable_node_list.size() << std::endl;
  spot_file << "***********************************" << std::endl;
  spot_file << std::endl;
  spot_file << std::endl;
  spot_file << std::endl;
  int cnt = 0;
  for (auto& node : uncolorable_node_list) {
    DrcPolygon* poly = node->get_polygon();
    if (poly != nullptr) {
      ++cnt;
      PolygonWithHoles poly_with_hole = poly->get_polygon();
      BoostRect bounding_box;
      bp::extents(bounding_box, poly_with_hole);
      auto left = static_cast<double>(bp::xl(bounding_box));
      auto bottom = static_cast<double>(bp::yl(bounding_box));
      auto right = static_cast<double>(bp::xh(bounding_box));
      auto top = static_cast<double>(bp::yh(bounding_box));
      spot_file << "uncolorable node " << cnt << " bounding box" << std::endl;
      spot_file << "Rect LeftBottom ::"
                << "(" << left / 1000 << "," << bottom / 1000 << ")"
                << " "
                << "Rect TopRight ::"
                << "(" << right / 1000 << "," << top / 1000 << ")" << std::endl;

      spot_file << std::endl;
    } else {
      std::cout << "nullptr !!!!!" << std::endl;
    }
  }
  spot_file.close();
}

}  // namespace idrc