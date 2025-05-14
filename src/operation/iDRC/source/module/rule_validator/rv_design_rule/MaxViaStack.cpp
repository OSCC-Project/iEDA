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
#include "RuleValidator.hpp"

namespace idrc {

void RuleValidator::verifyMaxViaStack(RVBox& rv_box)
{
  PropertyDefinition& property_definition = DRCDM.getDatabase().get_property_definition();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();

  // rule get
  // only suit for rule of :LEF58_MAXVIASTACK STRING "MAXVIASTACK 4 NOSINGLE RANGE M1 M7
  // it need modify
  int32_t max_via_stack = 4;
  int32_t metal_idx_start = 0;
  int32_t metal_idx_end = 6;
  // 上面三个应该是需要获取的。下面三个是需要推导的(也是下面代码真正用的数据)。 暂定写死，需要修改

  int32_t cut_idx_start = 1;
  int32_t cut_idx_end = 6;
  int32_t required_size = 8000;  // why 8000? 4*2000?

  std::map<int32_t, std::map<int32_t, std::vector<PlanarRect>>> cut_net_rect_map;
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> cut_bg_rtree_map;
  for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    if (drc_shape->get_is_routing()) {
      continue;
    }
    cut_net_rect_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
    cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
  }
  for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
    if (drc_shape->get_is_routing()) {
      continue;
    }
    cut_net_rect_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
    cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
  }

  for (auto& [cut_layer_idx, net_rect_map] : cut_net_rect_map) {
    // 从底往上查 查询layer为 [start,...,end - max_via_size](至少要连续max_via_size + 1个才会触发该违例)
    if (cut_layer_idx < cut_idx_start || cut_layer_idx > (cut_idx_end - max_via_stack)) {
      continue;
    }
    for (auto& [net_idx, rect_list] : net_rect_map) {
      for (PlanarRect& rect : rect_list) {
        int32_t via_stack_size = 1;

        std::map<int32_t, std::vector<std::pair<int32_t, PlanarRect>>> stack_layer_net_rect_map;  // 记录via stack的重叠信息
        stack_layer_net_rect_map[cut_layer_idx].push_back({net_idx, rect});
        for (int32_t cur_cut_layer_idx = cut_layer_idx + 1; cur_cut_layer_idx <= cut_idx_end; cur_cut_layer_idx++) {
          std::vector<std::pair<BGRectInt, int32_t>> bg_rect_net_pair_list;
          {
            // 用上一层所有rect构成一个外接矩形进行查找
            std::vector<PlanarRect> temp_rect_list;
            for (auto prev_layer_net_rect_pair : stack_layer_net_rect_map[cur_cut_layer_idx - 1]) {
              temp_rect_list.push_back(prev_layer_net_rect_pair.second);
            }
            PlanarRect check_rect = DRCUTIL.getBoundingBox(temp_rect_list);
            cut_bg_rtree_map[cur_cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_rect_net_pair_list));
          }
          if (bg_rect_net_pair_list.empty()) {
            break;
          }

          via_stack_size++;  // stack size增加

          for (auto& [bg_env_rect, env_net_idx] : bg_rect_net_pair_list) {
            bool isoverlap = false;
            // 被查询矩形中与上一层有相交的保留添加到这一层
            for (auto prev_layer_net_rect_pair : stack_layer_net_rect_map[cur_cut_layer_idx - 1]) {
              if (DRCUTIL.isOpenOverlap(prev_layer_net_rect_pair.second, DRCUTIL.convertToPlanarRect(bg_env_rect))) {
                isoverlap = true;
                break;
              }
            }
            if (isoverlap == false) {
              continue;
            }
            stack_layer_net_rect_map[cur_cut_layer_idx].push_back({env_net_idx, DRCUTIL.convertToPlanarRect(bg_env_rect)});
          }
        }

        if (via_stack_size <= max_via_stack) {  // 不超过max via 跳过
          continue;
        }
        // 依次获取违例矩形
        for (auto& [cut_layer_idx, net_rect_pair_list] : stack_layer_net_rect_map) {
          // 违例只会出现在start + max_via_size层到cut_idx_end层，用一个stack 最顶层的via作为违例矩形
          if (cut_layer_idx < cut_idx_start + max_via_stack || cut_layer_idx > cut_idx_end) {
            continue;
          }
          int32_t routing_layer_idx = -1;
          {
            std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
            routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
          }
          for (auto& [net_idx, rect] : net_rect_pair_list) {
            bool is_vio_cut = true;
            // 对于非最后一层的cut，能够被添加为违例的条件是与后面的layer的cut没有overlap，这样保证他没有在另一个via stack中
            if (DRCUTIL.exist(stack_layer_net_rect_map, cut_layer_idx + 1)) {
              for (auto post_layer_net_rect_pair : stack_layer_net_rect_map[cut_layer_idx - 1]) {
                if (DRCUTIL.isOpenOverlap(post_layer_net_rect_pair.second, rect)) {
                  is_vio_cut = false;
                  break;
                }
              }
            }
            if (is_vio_cut == false) {
              continue;
            }
            Violation violation;
            violation.set_violation_type(ViolationType::kMaxViaStack);
            violation.set_is_routing(true);
            violation.set_layer_idx(routing_layer_idx);
            violation.set_rect(rect);
            violation.set_violation_net_set({net_idx});
            violation.set_required_size(required_size);
            rv_box.get_violation_list().push_back(violation);
          }
        }
      }
    }
  }
}

}  // namespace idrc
