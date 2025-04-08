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
#include "Logger.hpp"
#include "RuleValidator.hpp"
#include "Utility.hpp"

namespace idrc {

void RuleValidator::verifySameLayerCutSpacing(RVBox& rv_box)
{
  auto getdiagonallength = [&](GTLRectInt rect) -> double {
    return bg::distance(bg::model::d2::point_xy<int>(gtl::xl(rect), gtl::yl(rect)), bg::model::d2::point_xy<int>(gtl::xh(rect), gtl::yh(rect)));
  };
  auto& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();

  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();

  std::map<int32_t, GTLPolySetInt> cut_gtl_poly_set_map;
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> cut_bg_rtree_map;

  // std::map<int32_t,std::vector<DRCShape*>>cut_net_id_to_drc_shape_map;
  std::map<int32_t, int32_t> cut_drc_shape_id_to_net_id_map;
  std::map<int32_t, DRCShape*> cut_drc_shape_id_to_drc_shape_map;

  int32_t cut_drc_shape_id = -1;

  for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    if (drc_shape->get_is_routing()) {
      continue;
    }

    cut_gtl_poly_set_map[drc_shape->get_layer_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());

    cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), ++cut_drc_shape_id));
    // cut_net_id_to_drc_shape_map[drc_shape->get_net_idx()].push_back(drc_shape);
    cut_drc_shape_id_to_net_id_map[cut_drc_shape_id] = drc_shape->get_net_idx();
    cut_drc_shape_id_to_drc_shape_map[cut_drc_shape_id] = drc_shape;
  }

  for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
    if (drc_shape->get_is_routing()) {
      continue;
    }
    cut_gtl_poly_set_map[drc_shape->get_layer_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
    cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), ++cut_drc_shape_id));
    // cut_net_id_to_drc_shape_map[drc_shape->get_net_idx()].push_back(drc_shape);
    cut_drc_shape_id_to_net_id_map[cut_drc_shape_id] = drc_shape->get_net_idx();
    cut_drc_shape_id_to_drc_shape_map[cut_drc_shape_id] = drc_shape;
  }

  for (auto& [cut_layer_idx, gtl_poly_set] : cut_gtl_poly_set_map) {
    int32_t prl_distance = -40*2;
    int32_t min_prl_spacing = cut_layer_list[cut_layer_idx].get_curr_prl_spacing();
    int32_t min_no_prl_spacing = cut_layer_list[cut_layer_idx].get_curr_spacing();

    int32_t min_spacing = min_prl_spacing;
    int32_t half_min_spacing = min_spacing / 2;

    int32_t routing_layer_idx = -1;
    {
      std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
      routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
      // // for debug missed
      // auto& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
      // auto& routing_layer = routing_layer_list[routing_layer_idx];
      // auto& routing_layer_name = routing_layer.get_layer_name();
      // if(routing_layer_name=="M2"){
      //   DRCLOG.info(Loc::current(), "routing_layer M2:");
      //   for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
      //     if (drc_shape->get_is_routing()) {
      //       continue;
      //     }
      //     DRCLOG.info(Loc::current(), "drc_shape: ",drc_shape->get_ll_x()/1000.0, ",", drc_shape->get_ll_y()/1000.0, "  ", drc_shape->get_ur_x()/1000.0, ",", drc_shape->get_ur_y()/1000.0);
      //   }
      // }

      //
    }
    std::vector<PlanarRect> violation_rect_list;
    {
      gtl_poly_set.clean();
      gtl::grow_and(gtl_poly_set, half_min_spacing);

      std::vector<GTLRectInt> vio_rects;
      gtl::get_max_rectangles(vio_rects,gtl_poly_set);

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
          if (getdiagonallength(vio_rects[i]) >= min_spacing) {
            mark_save[i] = false;
          }
        }
      }
      for (int i = 0; i < (int) vio_rects.size(); i++) {
        if (true == mark_save[i]) {
          std::set<int32_t> env_drc_shape_idx_set;
          {
            std::vector<std::pair<BGRectInt, int32_t>> bg_rect_drc_shape_id_pair_list;

            cut_bg_rtree_map[cut_layer_idx].query(
                bgi::intersects(DRCUTIL.convertToBGRectInt(DRCUTIL.getEnlargedRect(DRCUTIL.convertToPlanarRect(vio_rects[i]), 1))),
                std::back_inserter(bg_rect_drc_shape_id_pair_list));
            for (auto& [bg_rect, drc_shap_idx] : bg_rect_drc_shape_id_pair_list) {
              env_drc_shape_idx_set.insert(drc_shap_idx);
            }
          }
          if (env_drc_shape_idx_set.empty() || (env_drc_shape_idx_set.size() == 1 && *env_drc_shape_idx_set.begin() == -1)) {
            continue;
            // FIX:need to modify
          }
          // check corners functions
          auto check_line_intersect = [&](int32_t line1left, int32_t line1right, int32_t line2left, int32_t line2right) -> bool {
            if (line1left > line2right || line1right < line2left) {
              return false;
            }
            return true;
          };
          auto check_corner_to_corner = [&](GTLRectInt rect1, GTLRectInt rect2) -> bool {
            if (!check_line_intersect(gtl::xl(rect1), gtl::xh(rect1), gtl::xl(rect2), gtl::xh(rect2))
                && !check_line_intersect(gtl::yl(rect1), gtl::yh(rect1), gtl::yl(rect2), gtl::yh(rect2))) {
              return true;
            }
            return false;
          };
          auto check_corners = [&](DRCShape* shape1, DRCShape* shape2) -> bool {
            return check_corner_to_corner(DRCUTIL.convertToGTLRectInt(shape1->get_rect()), DRCUTIL.convertToGTLRectInt(shape2->get_rect()));
          };

          using point1d_t = bg::model::point<int32_t, 1, bg::cs::cartesian>;
          using segment1d_t = bg::model::segment<point1d_t>;
          // check prl functions
          auto get_prl_lenth = [](int32_t line1left, int32_t line1right, int32_t line2left, int32_t line2right) -> int32_t {
            // auto get_prl_lenth = [](segment1d_t segment1, segment1d_t segment2) -> int32_t {
            // intersection似乎不支出 1d的情况。
            // // 获取相交部分的长度
            // std::vector<segment1d_t> intersection;
            // bg::intersection(segment1, segment2, intersection);
            // if (intersection.size() != 1) {
            //   DRCLOG.info(Loc::current(), "intersection size is not 1");
            // }
            // if (intersection.size() == 0) {
            //   return -bg::distance(segment1, segment2);
            // }
            // if (intersection.size() == 1) {
            //   int32_t prl_length = 0;
            //   for (auto& seg : intersection) {
            //     prl_length += bg::length(seg);
            //   }
            //   return prl_length;
            // }
            // return 0;
            int32_t total_left = std::min(line1left, line2left);
            int32_t total_right = std::max(line1right, line2right);
            int32_t line1_length = line1right - line1left;
            int32_t line2_length = line2right - line2left;
            return line1_length + line2_length - (total_right - total_left);
          };
          auto check_2_direction_prl_lenth = [&](GTLRectInt rect1, GTLRectInt rect2, int32_t prl_distance) -> bool {
            // auto prl_length_horizontal = get_prl_lenth(segment1d_t(point1d_t(gtl::xl(rect1)), point1d_t(gtl::xh(rect1))),
            //                                            segment1d_t(point1d_t(gtl::xl(rect2)), point1d_t(gtl::xh(rect2))));
            // auto prl_length_vertical = get_prl_lenth(segment1d_t(point1d_t(gtl::yl(rect1)), point1d_t(gtl::yh(rect1))),
            //                                          segment1d_t(point1d_t(gtl::yl(rect2)), point1d_t(gtl::yh(rect2))));
            auto prl_length_horizontal = get_prl_lenth(gtl::xl(rect1), gtl::xh(rect1), gtl::xl(rect2), gtl::xh(rect2));
            auto prl_length_vertical = get_prl_lenth(gtl::yl(rect1), gtl::yh(rect1), gtl::yl(rect2), gtl::yh(rect2));
            DRCLOG.info(Loc::current(), "prl_length_horizontal: ", prl_length_horizontal);
            DRCLOG.info(Loc::current(), "prl_length_vertical: ", prl_length_vertical);
            return (prl_length_horizontal > prl_distance || prl_length_vertical > prl_distance);
          };

          auto check_prl = [&](DRCShape* shape1, DRCShape* shape2, int32_t prl_distance) -> bool {
            return check_2_direction_prl_lenth(DRCUTIL.convertToGTLRectInt(shape1->get_rect()), DRCUTIL.convertToGTLRectInt(shape2->get_rect()), prl_distance);
          };

          std::vector<std::pair<std::set<int32_t>, bool>> net_idx_set_prl_list;

          {
            
            std::vector<int32_t> drc_shape_idx_list(env_drc_shape_idx_set.begin(), env_drc_shape_idx_set.end());
            for (size_t _i = 0; _i < drc_shape_idx_list.size(); ++_i) {
              for (size_t _j = _i + 1; _j < drc_shape_idx_list.size(); ++_j) {
                int32_t env_drc_shape_idx1 = drc_shape_idx_list[_i];
                int32_t env_drc_shape_idx2 = drc_shape_idx_list[_j];
                bool is_corner_to_corner
                    = check_corners(cut_drc_shape_id_to_drc_shape_map[env_drc_shape_idx1], cut_drc_shape_id_to_drc_shape_map[env_drc_shape_idx2]);
                bool is_prl
                    = check_prl(cut_drc_shape_id_to_drc_shape_map[env_drc_shape_idx1], cut_drc_shape_id_to_drc_shape_map[env_drc_shape_idx2], prl_distance);
                auto shape1 = cut_drc_shape_id_to_drc_shape_map[env_drc_shape_idx1];
                auto shape2 = cut_drc_shape_id_to_drc_shape_map[env_drc_shape_idx2];
                // DRCLOG.info(Loc::current(), "shape1: ",shape1->get_ll_x()/1000.0, ",", shape1->get_ll_y()/1000.0, "  ", shape1->get_ur_x()/1000.0, ",", shape1->get_ur_y()/1000.0);
                // DRCLOG.info(Loc::current(), "shape2: ",shape2->get_ll_x()/1000.0, ",", shape2->get_ll_y()/1000.0, "  ", shape2->get_ur_x()/1000.0, ",", shape2->get_ur_y()/1000.0);
                // DRCLOG.info(Loc::current(), "vio_rect: ", DRCUTIL.convertToPlanarRect(vio_rects[i]).get_ll_x()/1000.0, ",", DRCUTIL.convertToPlanarRect(vio_rects[i]).get_ll_y()/1000.0, "  ",
                //             DRCUTIL.convertToPlanarRect(vio_rects[i]).get_ur_x()/1000.0, ",", DRCUTIL.convertToPlanarRect(vio_rects[i]).get_ur_y()/1000.0);  
                // if (is_prl) {
                //       DRCLOG.info(Loc::current(), "prl violation");
                //     } else {
                //       DRCLOG.info(Loc::current(), "no prl violation");
                //     }
                
                // 治标不治本:重新获取vio_rect。解决gtl_poly_set.get(vio_rects);的切分问题
                // 处理时机不好定。
                // auto vio_rect_new_bloat=DRCUTIL.getOverlap(DRCUTIL.getEnlargedRect(cut_drc_shape_id_to_drc_shape_map[env_drc_shape_idx1]->get_rect(),half_min_spacing),
                //                                      DRCUTIL.getEnlargedRect(cut_drc_shape_id_to_drc_shape_map[env_drc_shape_idx2]->get_rect(),half_min_spacing));
                // auto vio_rect_new=DRCUTIL.getShrinkedRect(vio_rect_new_bloat,half_min_spacing);
                // vio_rects[i]=DRCUTIL.convertToGTLRectInt(vio_rect_new);
                auto bg_rect_vio = DRCUTIL.convertToBGRectInt(vio_rects[i]);
                if (is_corner_to_corner == false || (is_corner_to_corner && DRCUTIL.getDiagonalLength(bg_rect_vio) < min_spacing)) {
                  // if(is_corner_to_corner==false||(is_corner_to_corner && getdiagonallength(vio_rects[i])<min_spacing)){
                  // FIX: here use float type ,there will be more bugs
                  net_idx_set_prl_list.push_back(std::pair<std::set<int32_t>, bool>{
                      {cut_drc_shape_id_to_net_id_map[env_drc_shape_idx1], cut_drc_shape_id_to_net_id_map[env_drc_shape_idx2]}, is_prl});
                }
              }
            }
          }
          for (auto [net_idx_set, is_prl] : net_idx_set_prl_list) {
            Violation violation;
            violation.set_violation_type(ViolationType::kSameLayerCutSpacing);
            // violation.set_required_size(min_spacing);
            violation.set_required_size(is_prl ? min_prl_spacing : min_no_prl_spacing);
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
