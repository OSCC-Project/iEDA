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
 * @File Name: contest_evaluation.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "contest_evaluation.h"

#include <iostream>

#include "contest_dm.h"

namespace ieda_contest {

ContestEvaluation::ContestEvaluation(ContestDataManager* data_manager)
{
  _data_manager = data_manager;

  // 构建评估需要的辅助数据


}

bool ContestEvaluation::doEvaluation(std::string report_file)
{
  // overlap检查
  if (!overlapCheckPassed()) {
    std::cout << "Overlap check failed!" << std::endl;
    return false;
  }

  // 连通性检查
  if (!connectivityCheckPassed()) {
    std::cout << "Connectivity check failed!" << std::endl;
    return false;
  }

  // overflow检查
  if (!overflowCheckPassed()) {
    std::cout << "Overflow check failed!" << std::endl;
    return false;
  }

  double score = 0;
  // 计算时序分数
  score += calcTimingScore();
  std::cout << "##############################" << std::endl;
  std::cout << "Final score: " << score << std::endl;
  std::cout << "##############################" << std::endl;

  return true;
}

bool ContestEvaluation::overlapCheckPassed()
{
  return true;
}

bool ContestEvaluation::connectivityCheckPassed()
{
  return true;
}

// #if 1

// // 从segment_list 到 tree的完全流程 (包括构建 优化 检查)
// static MTree<LayerCoord> getTreeByFullFlow(std::vector<LayerCoord>& candidate_root_coord_list,
//                                            std::vector<Segment<LayerCoord>>& segment_list,
//                                            std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map)
// {
//   // 判断是否有斜线段
//   if (!passCheckingOblique(segment_list)) {
//     LOG_INST.error(Loc::current(), "There are oblique segments in segment_list!");
//   }
//   // 删除点线段
//   erasePointSegment(segment_list);
//   // 融合重叠的线段
//   mergeOverlapSegment(segment_list, key_coord_pin_map);
//   // 从候选的root坐标列表中得到树root结点
//   LayerCoord root_value = getRootCoord(candidate_root_coord_list, segment_list, key_coord_pin_map);
//   // 构建坐标树
//   MTree<LayerCoord> coord_tree = getTreeBySegList(root_value, segment_list);
//   // 删除无效(没有关键坐标的子树)的结点
//   eraseInvalidNode(coord_tree, key_coord_pin_map);
//   // 融合中间(平面进行切点融合,通孔进行层切割)结点
//   mergeMiddleNode(coord_tree, key_coord_pin_map);
//   // 检查树中是否有斜线
//   if (!passCheckingOblique(coord_tree)) {
//     LOG_INST.error(Loc::current(), "There are oblique segments in tree!");
//   }
//   // 检查树是否到达所有的关键坐标
//   if (!passCheckingReachable(coord_tree, key_coord_pin_map)) {
//     LOG_INST.error(Loc::current(), "The key points unreachable!");
//   }
//   return coord_tree;
// }

// // 判断是否有斜线
// static bool passCheckingOblique(std::vector<Segment<LayerCoord>>& segment_list)
// {
//   for (Segment<LayerCoord>& segment : segment_list) {
//     Orientation orientation = getOrientation(segment.get_first(), segment.get_second());
//     if (orientation == Orientation::kOblique) {
//       return false;
//     }
//   }
//   return true;
// }

// // 删除点线段
// static void erasePointSegment(std::vector<Segment<LayerCoord>>& segment_list)
// {
//   std::vector<Segment<LayerCoord>> new_segment_list;
//   for (Segment<LayerCoord>& segment : segment_list) {
//     if (segment.get_first() == segment.get_second()) {
//       continue;
//     }
//     new_segment_list.push_back(segment);
//   }
//   segment_list = new_segment_list;
// }

// // 融合重叠的线段
// static void mergeOverlapSegment(std::vector<Segment<LayerCoord>>& segment_list,
//                                 std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map)
// {
//   std::vector<Segment<LayerCoord>> h_segment_list;
//   std::vector<Segment<LayerCoord>> v_segment_list;
//   std::vector<Segment<LayerCoord>> p_segment_list;

//   for (Segment<LayerCoord>& segment : segment_list) {
//     if (isHorizontal(segment.get_first(), segment.get_second())) {
//       h_segment_list.push_back(segment);
//     } else if (isVertical(segment.get_first(), segment.get_second())) {
//       v_segment_list.push_back(segment);
//     } else if (isProximal(segment.get_first(), segment.get_second())) {
//       p_segment_list.push_back(segment);
//     }
//   }
//   // 先切柱子
//   std::vector<Segment<LayerCoord>> p_segment_list_temp;
//   for (Segment<LayerCoord>& p_segment : p_segment_list) {
//     PlanarCoord& planar_coord = p_segment.get_first().get_planar_coord();
//     irt_int first_layer_idx = p_segment.get_first().get_layer_idx();
//     irt_int second_layer_idx = p_segment.get_second().get_layer_idx();
//     swapASC(first_layer_idx, second_layer_idx);
//     for (irt_int layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
//       p_segment_list_temp.emplace_back(LayerCoord(planar_coord, layer_idx), LayerCoord(planar_coord, layer_idx + 1));
//     }
//   }
//   p_segment_list = p_segment_list_temp;

//   // 初始化平面切点
//   std::map<irt_int, std::set<irt_int>> x_cut_list_map;
//   std::map<irt_int, std::set<irt_int>> y_cut_list_map;

//   for (Segment<LayerCoord>& h_segment : h_segment_list) {
//     LayerCoord& first_coord = h_segment.get_first();
//     LayerCoord& second_coord = h_segment.get_second();
//     irt_int layer_idx = first_coord.get_layer_idx();

//     x_cut_list_map[layer_idx].insert(first_coord.get_x());
//     x_cut_list_map[layer_idx].insert(second_coord.get_x());
//     y_cut_list_map[layer_idx].insert(first_coord.get_y());
//   }
//   for (Segment<LayerCoord>& v_segment : v_segment_list) {
//     LayerCoord& first_coord = v_segment.get_first();
//     LayerCoord& second_coord = v_segment.get_second();
//     irt_int layer_idx = first_coord.get_layer_idx();

//     y_cut_list_map[layer_idx].insert(first_coord.get_y());
//     y_cut_list_map[layer_idx].insert(second_coord.get_y());
//     x_cut_list_map[layer_idx].insert(first_coord.get_x());
//   }
//   for (Segment<LayerCoord>& p_segment : p_segment_list) {
//     LayerCoord& first_coord = p_segment.get_first();
//     irt_int first_layer_idx = first_coord.get_layer_idx();

//     LayerCoord& second_coord = p_segment.get_second();
//     irt_int second_layer_idx = second_coord.get_layer_idx();

//     x_cut_list_map[first_layer_idx].insert(first_coord.get_x());
//     y_cut_list_map[first_layer_idx].insert(first_coord.get_y());
//     x_cut_list_map[second_layer_idx].insert(second_coord.get_x());
//     y_cut_list_map[second_layer_idx].insert(second_coord.get_y());
//   }
//   for (auto& [key_coord, pin_idx] : key_coord_pin_map) {
//     irt_int layer_idx = key_coord.get_layer_idx();
//     x_cut_list_map[layer_idx].insert(key_coord.get_x());
//     y_cut_list_map[layer_idx].insert(key_coord.get_y());
//   }

//   // 切割平面的h线
//   std::vector<Segment<LayerCoord>> h_segment_list_temp;
//   for (Segment<LayerCoord>& h_segment : h_segment_list) {
//     irt_int first_x = h_segment.get_first().get_x();
//     irt_int second_x = h_segment.get_second().get_x();
//     irt_int y = h_segment.get_first().get_y();
//     irt_int layer_idx = h_segment.get_first().get_layer_idx();

//     swapASC(first_x, second_x);
//     std::vector<irt_int> x_list;
//     for (irt_int x_cut : x_cut_list_map[layer_idx]) {
//       if (first_x <= x_cut && x_cut <= second_x) {
//         x_list.push_back(x_cut);
//       }
//     }
//     for (size_t i = 1; i < x_list.size(); i++) {
//       LayerCoord first_coord(x_list[i - 1], y, layer_idx);
//       LayerCoord second_coord(x_list[i], y, layer_idx);
//       h_segment_list_temp.emplace_back(first_coord, second_coord);
//     }
//   }
//   h_segment_list = h_segment_list_temp;

//   // 切割平面的v线
//   std::vector<Segment<LayerCoord>> v_segment_list_temp;
//   for (Segment<LayerCoord>& v_segment : v_segment_list) {
//     irt_int first_y = v_segment.get_first().get_y();
//     irt_int second_y = v_segment.get_second().get_y();
//     irt_int x = v_segment.get_first().get_x();
//     irt_int layer_idx = v_segment.get_first().get_layer_idx();

//     swapASC(first_y, second_y);
//     std::vector<irt_int> y_list;
//     for (irt_int y_cut : y_cut_list_map[layer_idx]) {
//       if (first_y <= y_cut && y_cut <= second_y) {
//         y_list.push_back(y_cut);
//       }
//     }
//     for (size_t i = 1; i < y_list.size(); i++) {
//       LayerCoord first_coord(x, y_list[i - 1], layer_idx);
//       LayerCoord second_coord(x, y_list[i], layer_idx);
//       v_segment_list_temp.emplace_back(first_coord, second_coord);
//     }
//   }
//   v_segment_list = v_segment_list_temp;

//   auto mergeSegmentList = [](std::vector<Segment<LayerCoord>>& segment_list) {
//     for (Segment<LayerCoord>& segment : segment_list) {
//       SortSegmentInnerXASC()(segment);
//     }
//     std::sort(segment_list.begin(), segment_list.end(), CmpSegmentXASC());
//     RTUtil::merge(segment_list, [](Segment<LayerCoord>& sentry, Segment<LayerCoord>& soldier) {
//       return (sentry.get_first() == soldier.get_first()) && (sentry.get_second() == soldier.get_second());
//     });
//   };
//   mergeSegmentList(h_segment_list);
//   mergeSegmentList(v_segment_list);
//   mergeSegmentList(p_segment_list);

//   segment_list.clear();
//   segment_list.insert(segment_list.end(), h_segment_list.begin(), h_segment_list.end());
//   segment_list.insert(segment_list.end(), v_segment_list.begin(), v_segment_list.end());
//   segment_list.insert(segment_list.end(), p_segment_list.begin(), p_segment_list.end());
// }

// // 从候选的root坐标列表中得到树root结点
// static LayerCoord getRootCoord(std::vector<LayerCoord>& candidate_root_coord_list, std::vector<Segment<LayerCoord>>& segment_list,
//                                std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map)
// {
//   LayerCoord root_coord;

//   for (Segment<LayerCoord>& segment : segment_list) {
//     for (LayerCoord& candidate_root_coord : candidate_root_coord_list) {
//       if (!isInside(segment, candidate_root_coord)) {
//         continue;
//       }
//       return candidate_root_coord;
//     }
//   }
//   if (!segment_list.empty()) {
//     LOG_INST.error(Loc::current(), "The segment_list not covered driving_pin!");
//   }
//   irt_int max_pin_num = INT32_MIN;
//   for (auto& [key_coord, pin_idx_set] : key_coord_pin_map) {
//     irt_int pin_num = static_cast<irt_int>(pin_idx_set.size());
//     if (max_pin_num < pin_num) {
//       max_pin_num = pin_num;
//       root_coord = key_coord;
//     }
//   }
//   if (max_pin_num == INT32_MIN) {
//     root_coord = candidate_root_coord_list.front();
//   }
//   return root_coord;
// }

// // 删除无效(没有关键坐标的子树)的结点
// static void eraseInvalidNode(MTree<LayerCoord>& coord_tree, std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>&
// key_coord_pin_map)
// {
//   std::vector<TNode<LayerCoord>*> erase_node_list;
//   std::map<TNode<LayerCoord>*, TNode<LayerCoord>*> curr_to_parent_node_map;
//   std::queue<TNode<LayerCoord>*> node_queue = initQueue(coord_tree.get_root());
//   while (!node_queue.empty()) {
//     TNode<LayerCoord>* node = getFrontAndPop(node_queue);
//     std::vector<TNode<LayerCoord>*> child_list = node->get_child_list();
//     addListToQueue(node_queue, child_list);

//     for (TNode<LayerCoord>* child_node : child_list) {
//       curr_to_parent_node_map[child_node] = node;

//       if (child_node->isLeafNode() && !exist(key_coord_pin_map, child_node->value())) {
//         erase_node_list.push_back(child_node);
//         TNode<LayerCoord>* parent_node = curr_to_parent_node_map[child_node];
//         parent_node->delChild(child_node);

//         while (parent_node->isLeafNode() && !exist(key_coord_pin_map, parent_node->value())) {
//           erase_node_list.push_back(parent_node);
//           TNode<LayerCoord>* child_node = parent_node;
//           parent_node = curr_to_parent_node_map[parent_node];
//           parent_node->delChild(child_node);
//         }
//       }
//     }
//   }
//   for (TNode<LayerCoord>* erase_node : erase_node_list) {
//     delete erase_node;
//   }
// }

// // 融合中间(平面进行切点融合,通孔进行层切割)结点
// static void mergeMiddleNode(MTree<LayerCoord>& coord_tree, std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>&
// key_coord_pin_map)
// {
//   std::vector<TNode<LayerCoord>*> merge_node_list;
//   std::map<TNode<LayerCoord>*, TNode<LayerCoord>*> middle_to_start_node_map;
//   std::queue<TNode<LayerCoord>*> node_queue = initQueue(coord_tree.get_root());
//   while (!node_queue.empty()) {
//     TNode<LayerCoord>* node = getFrontAndPop(node_queue);
//     addListToQueue(node_queue, node->get_child_list());
//     irt_int node_layer_idx = node->value().get_layer_idx();
//     PlanarCoord& node_coord = node->value().get_planar_coord();

//     for (TNode<LayerCoord>* child_node : node->get_child_list()) {
//       irt_int child_node_layer_idx = child_node->value().get_layer_idx();
//       PlanarCoord& child_node_coord = child_node->value().get_planar_coord();

//       if (node_layer_idx == child_node_layer_idx && node_coord != child_node_coord) {
//         middle_to_start_node_map[child_node] = node;
//         if (!exist(middle_to_start_node_map, node)) {
//           continue;
//         }
//         TNode<LayerCoord>* parent_node = middle_to_start_node_map[node];
//         if (getDirection(parent_node->value().get_planar_coord(), node_coord) == getDirection(node_coord, child_node_coord)
//             && node->getChildrenNum() == 1 && !exist(key_coord_pin_map, node->value())) {
//           parent_node->delChild(node);
//           parent_node->addChild(child_node);
//           merge_node_list.push_back(node);
//           middle_to_start_node_map[child_node] = parent_node;
//         }
//       }
//     }
//   }
//   for (TNode<LayerCoord>* merge_node : merge_node_list) {
//     delete merge_node;
//   }
// }

// // 检查树中是否有斜线
// static bool passCheckingOblique(MTree<LayerCoord>& coord_tree)
// {
//   for (TNode<LayerCoord>* coord_node : getNodeList(coord_tree)) {
//     LayerCoord& coord = coord_node->value();

//     PlanarCoord& first_planar_coord = coord.get_planar_coord();
//     irt_int first_layer_idx = coord.get_layer_idx();
//     PlanarCoord& second_planar_coord = coord.get_planar_coord();
//     irt_int second_layer_idx = coord.get_layer_idx();

//     if (first_layer_idx == second_layer_idx && isRightAngled(first_planar_coord, second_planar_coord)) {
//       continue;
//     } else if (first_layer_idx != second_layer_idx && first_planar_coord == second_planar_coord) {
//       continue;
//     }
//     return false;
//   }
//   return true;
// }

// // 检查树是否到达所有的关键坐标
// static bool passCheckingReachable(MTree<LayerCoord>& coord_tree,
//                                   std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map)
// {
//   std::map<irt_int, bool> visited_map;
//   for (auto& [key_coord, pin_idx_list] : key_coord_pin_map) {
//     for (irt_int pin_idx : pin_idx_list) {
//       visited_map[pin_idx] = false;
//     }
//   }
//   for (TNode<LayerCoord>* coord_node : getNodeList(coord_tree)) {
//     LayerCoord coord = coord_node->value();
//     if (!exist(key_coord_pin_map, coord)) {
//       continue;
//     }
//     for (irt_int pin_idx : key_coord_pin_map[coord]) {
//       visited_map[pin_idx] = true;
//     }
//   }
//   for (auto [pin_idx, is_visited] : visited_map) {
//     if (is_visited == false) {
//       LOG_INST.warning(Loc::current(), "The pin idx ", pin_idx, " unreachable!");
//       return false;
//     }
//   }
//   return true;
// }

// #endif

bool ContestEvaluation::overflowCheckPassed()
{
  return true;
}

double ContestEvaluation::calcTimingScore()
{
  return 101;
}

}  // namespace ieda_contest
