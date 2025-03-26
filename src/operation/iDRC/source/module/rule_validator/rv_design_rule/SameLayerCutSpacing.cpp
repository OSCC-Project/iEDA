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

void RuleValidator::verifySameLayerCutSpacing(RVBox& rv_box)
{
  auto& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();

  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();

  std::map<int32_t, GTLPolySetInt> cut_gtl_poly_set_map;
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> cut_bg_rtree_map;

  for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    if (drc_shape->get_is_routing()) {
      continue;
    }

    cut_gtl_poly_set_map[drc_shape->get_layer_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
    cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
  }

  for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
    if (drc_shape->get_is_routing()) {
      continue;
    }
    cut_gtl_poly_set_map[drc_shape->get_layer_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
    cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
  }

  for (auto& [cut_layer_idx, gtl_poly_set] : cut_gtl_poly_set_map) {
    int32_t min_spacing = cut_layer_list[cut_layer_idx].get_curr_spacing();
    int32_t half_min_spacing = min_spacing / 2;

    int32_t routing_layer_idx = -1;
    {
      std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
      routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    }
    std::vector<PlanarRect> violation_rect_list;
    {
      gtl_poly_set.clean();
      gtl::grow_and(gtl_poly_set, half_min_spacing);

      std::vector<GTLRectInt> vio_rects;
      gtl_poly_set.get(vio_rects);

      std::vector<bool> mark_save(vio_rects.size(), true);

      auto get_new_interval = [&](gtl::orientation_2d direction, GTLRectInt& rect) -> bool {
        int length = gtl::delta(rect, direction);
        if (length <= half_min_spacing) {
          int expand_length = std::abs(half_min_spacing - length);
          gtl::bloat(rect, direction, expand_length);
          return true;
        } else if (length > min_spacing) {
          int shrink_length = (length - min_spacing) == 0 ? half_min_spacing - 1 : half_min_spacing;
          gtl::shrink(rect, direction, shrink_length);
          return false;
        } else {
          int shrink_length = std::abs(half_min_spacing - length);
          gtl::shrink(rect, direction, shrink_length);
          return false;
        }
      };
      int cut_sapcing_violation_num = 0;
      for (int i = 0; i < (int) vio_rects.size(); i++) {
        auto state_h = get_new_interval(gtl::HORIZONTAL, vio_rects[i]);
        auto state_v = get_new_interval(gtl::VERTICAL, vio_rects[i]);
        if (state_h && state_v) {
          if (bg::distance(bg::model::d2::point_xy<int>(gtl::xl(vio_rects[i]), gtl::yl(vio_rects[i])),
                           bg::model::d2::point_xy<int>(gtl::xh(vio_rects[i]), gtl::yh(vio_rects[i])))
              >= min_spacing) {
            mark_save[i] = false;
          }
        }
      }
      for (int i = 0; i < (int) vio_rects.size(); i++) {
        if (true == mark_save[i]) {
          std::set<int32_t> env_net_idx_set;
          {
            std::vector<std::pair<BGRectInt, int32_t>> bg_rect_net_pair_list;

            cut_bg_rtree_map[cut_layer_idx].query(
                bgi::intersects(DRCUTIL.convertToBGRectInt(DRCUTIL.getEnlargedRect(DRCUTIL.convertToPlanarRect(vio_rects[i]), 1))),
                std::back_inserter(bg_rect_net_pair_list));
            for (auto& [bg_rect, net_idx] : bg_rect_net_pair_list) {
              env_net_idx_set.insert(net_idx);
            }
          }
          if (env_net_idx_set.empty() || (env_net_idx_set.size() == 1 && *env_net_idx_set.begin() == -1)) {
            continue;
          }
          std::vector<std::set<int32_t>> net_idx_set_list;
          {
            if (env_net_idx_set.size() <= 2) {
              net_idx_set_list.push_back(env_net_idx_set);
            } else {
              std::vector<int32_t> net_idx_list(env_net_idx_set.begin(), env_net_idx_set.end());
              for (size_t i = 0; i < net_idx_list.size(); ++i) {
                for (size_t j = i + 1; j < net_idx_list.size(); ++j) {
                  net_idx_set_list.emplace_back(std::set<int32_t>{net_idx_list[i], net_idx_list[j]});
                }
              }
            }
          }
          for (std::set<int32_t>& net_idx_set : net_idx_set_list) {
            Violation violation;
            violation.set_violation_type(ViolationType::kSameLayerCutSpacing);
            violation.set_required_size(min_spacing);
            violation.set_is_routing(true);
            violation.set_violation_net_set(net_idx_set);
            violation.set_layer_idx(routing_layer_idx);
            violation.set_rect(DRCUTIL.getShrinkedRect(DRCUTIL.convertToPlanarRect(vio_rects[i]), 0));
            rv_box.get_violation_list().push_back(violation);
            cut_sapcing_violation_num++;
          }
        }
      }
      // DRCLOG.info(Loc::current(), "cut sapcing violation num: ", cut_sapcing_violation_num);
    }
  }
}

}  // namespace idrc
