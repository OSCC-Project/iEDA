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

void RuleValidator::verifyEndOfLineSpacing(RVBox& rv_box)
{
  // eol spacing 包括的东西
  struct EOLRule
  {
    int32_t eol_spacing;
    int32_t eol_width;
    int32_t eol_within;
    bool has_ete;
    int32_t ete_space;

    bool has_par;
    bool has_sub_trace_eol_width;
    int32_t par_space;
    int32_t par_within;
    bool has_two_edges;

    bool has_min_length;
    int32_t min_length;

    bool has_enclose_cut;
    bool has_below;
    bool has_above;
    int32_t enclosed_dist;
    int32_t cut_to_metal_space;
    bool has_all_cuts;

    bool has_same_metal;
  };

  // define func
  auto get_polygon_info = [&](GTLHolePolyInt& gtl_hole_poly, int32_t& coord_size, std::vector<PlanarCoord>& coord_list, std::vector<bool>& convex_corner_list,
                              std::vector<int32_t>& edge_length_list, std::vector<Segment<PlanarCoord>>& edge_list, std::set<int32_t>& eol_edge_idx_set) {
    for (auto iter = gtl_hole_poly.begin(); iter != gtl_hole_poly.end(); iter++) {
      coord_list.push_back(DRCUTIL.convertToPlanarCoord(*iter));
    }
    for (int32_t i = 0; i < coord_size; i++) {
      PlanarCoord& pre_coord = coord_list[getIdx(i - 1, coord_size)];
      PlanarCoord& curr_coord = coord_list[i];
      PlanarCoord& post_coord = coord_list[getIdx(i + 1, coord_size)];
      convex_corner_list.push_back(DRCUTIL.isConvexCorner(DRCUTIL.getRotation(gtl_hole_poly), pre_coord, curr_coord, post_coord));
      edge_length_list.push_back(DRCUTIL.getManhattanDistance(pre_coord, curr_coord));
      edge_list.push_back(Segment<PlanarCoord>(pre_coord, curr_coord));
    }

    for (int32_t i = 0; i < coord_size; i++) {
      if (convex_corner_list[getIdx(i - 1, coord_size)] && convex_corner_list[i]) {
        eol_edge_idx_set.insert(i);
      }
    }
  };

  // 构建图形，包括routing 层和cut层
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> cut_bg_rtree_map;
  // routing 层的信息
  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_gtl_poly_set_map;
  std::map<int32_t, std::map<int32_t, std::vector<GTLHolePolyInt>>> layer_net_polygon_list;  // 存储所有polygon的map
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, std::pair<int32_t, int32_t>>, bgi::quadratic<16>>> routing_bg_rtree_map;

  for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    if (!drc_shape->get_is_routing()) {
      cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
    } else
      routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
  }
  for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
    if (!drc_shape->get_is_routing()) {
      cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
    } else
      routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
  }

  for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_gtl_poly_set_map) {
    for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
      std::vector<GTLHolePolyInt> gtl_hole_poly_list;
      gtl_poly_set.get(gtl_hole_poly_list);
      for (GTLHolePolyInt& gtl_hole_poly : gtl_hole_poly_list) {
        layer_net_polygon_list[routing_layer_idx][net_idx].push_back(gtl_hole_poly);
        std::vector<GTLRectInt> gtl_rect_list;
        gtl::get_max_rectangles(gtl_rect_list, gtl_hole_poly);
        for (GTLRectInt& gtl_rect : gtl_rect_list) {
          // 存储max rect和他所对应的polygon的信息(polygon 的索引)
          routing_bg_rtree_map[routing_layer_idx].insert(
              std::make_pair(DRCUTIL.convertToBGRectInt(gtl_rect), std::make_pair(net_idx, layer_net_polygon_list[routing_layer_idx][net_idx].size() - 1)));
        }
      }
    }
  }
  std::vector<EOLRule> eol_rule_list;
  eol_rule_list.push_back({/*eol spacing*/ 140, /*eol width*/ 140, /*eol within*/ 50, /*ete*/ true, 160,
                           /*has par*/ false, /*sub_eol*/ false, /*par space*/ 0, /*par within*/ 0, /* two egde*/ false, /*min length*/ false, 0,
                           /*enclose cut*/ false, /*below */ false, /*above */ false, /*enclose dist */ 0, /*cut to metal space*/ 0, /*all cuts*/ false,
                           /*same metal*/ false});
  eol_rule_list.push_back({/*eol spacing*/ 160, /*eol width*/ 140, /*eol within*/ 50, /*ete*/ true, 160,
                           /*has par*/ true, /*sub_eol*/ true, /*par space*/ 230, /*par within*/ 140, /* two egde*/ false, /*min length*/ true, 100,
                           /*enclose cut*/ false, /*below */ false, /*above */ false, /*enclose dist */ 0, /*cut to metal space*/ 0, /*all cuts*/ false,
                           /*same metal*/ false});
  eol_rule_list.push_back({/*eol spacing*/ 200, /*eol width*/ 140, /*eol within*/ 50, /*ete*/ true, 160,
                           /*has par*/ true, /*sub_eol*/ true, /*par space*/ 230, /*par within*/ 140, /* two egde*/ false, /*min length*/ true, 100,
                           /*enclose cut*/ true, /*below */ true, /*above */ false, /*enclose dist */ 100, /*cut to metal space*/ 290, /*all cuts*/ true,
                           /*same metal*/ false});
  eol_rule_list.push_back({/*eol spacing*/ 230, /*eol width*/ 110, /*eol within*/ 0, /*ete*/ false, 0,
                           /*has par*/ true, /*sub_eol*/ false, /*par space*/ 120, /*par within*/ 240, /* two egde*/ true, /*min length*/ true, 300,
                           /*enclose cut*/ false, /*below */ false, /*above */ false, /*enclose dist */ 0, /*cut to metal space*/ 0, /*all cuts*/ false,
                           /*same metal*/ true});

  // 以polygon为单位进行eol的判定
  for (auto& [routing_layer_idx, net_polygon_list] : layer_net_polygon_list) {
    // 目前只管M2-M7的，也就是idx在[1,6]
    if (routing_layer_idx < 1 || routing_layer_idx > 6) {
      continue;
    }
    RoutingLayer& routing_layer = routing_layer_list[routing_layer_idx];

    // 处理一些规则项目
    bool use_cut = false;
    int32_t max_eol_width = eol_rule_list[0].eol_width;

    for (const EOLRule& eol_rule : eol_rule_list) {
      if (eol_rule.has_enclose_cut && eol_rule.has_below) {  // 目前只管below的
        use_cut = true;
      }
      max_eol_width = std::max(eol_rule.eol_width, max_eol_width);
    }
    struct SegmentLess
    {
      bool operator()(const Segment<PlanarCoord>& a, const Segment<PlanarCoord>& b) const
      {
        if (a.get_first().get_x() != b.get_first().get_x()) {
          return a.get_first().get_x() < b.get_first().get_x();
        }
        if (a.get_first().get_y() != b.get_first().get_y()) {
          return a.get_first().get_y() < b.get_first().get_y();
        }
        if (a.get_second().get_x() != b.get_second().get_x()) {
          return a.get_second().get_x() < b.get_second().get_x();
        }
        return a.get_second().get_y() < b.get_second().get_y();
      }
    };
    std::map<Segment<PlanarCoord>, std::vector<std::pair<Segment<PlanarCoord>, Violation>>, SegmentLess> edge_vio_map;
    for (auto& [net_idx, gtl_hole_poly_list] : net_polygon_list) {
      // for (GTLHolePolyInt& gtl_hole_poly : gtl_hole_poly_list) {
      for (int32_t gtl_hole_poly_idx = 0; gtl_hole_poly_idx < gtl_hole_poly_list.size(); gtl_hole_poly_idx++) {
        GTLHolePolyInt& gtl_hole_poly = gtl_hole_poly_list[gtl_hole_poly_idx];
        int32_t coord_size = static_cast<int32_t>(gtl_hole_poly.size());
        if (coord_size < 4) {
          continue;
        }

        // 拿到polygon info
        std::vector<PlanarCoord> coord_list;
        std::vector<bool> convex_corner_list;
        std::vector<int32_t> edge_length_list;
        std::set<int32_t> eol_edge_idx_set;
        std::vector<Segment<PlanarCoord>> edge_list;
        get_polygon_info(gtl_hole_poly, coord_size, coord_list, convex_corner_list, edge_length_list, edge_list, eol_edge_idx_set);
        // polygon的方向
        Rotation rotation = DRCUTIL.getRotation(gtl_hole_poly);

        // 拿到cut信息，先判断有没有规则用到cut
        std::vector<PlanarRect> polygon_cut_list;
        std::vector<GTLRectInt> polygon_gtl_rect_list;
        gtl::get_max_rectangles(polygon_gtl_rect_list, gtl_hole_poly);
        if (use_cut) {
          for (GTLRectInt& gtl_rect : polygon_gtl_rect_list) {
            std::vector<std::pair<BGRectInt, int32_t>> cut_bg_rect_net_pair_list;
            {
              int32_t cut_layer_idx = routing_layer_idx;  // 下层的刚好对应上
              cut_bg_rtree_map[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(gtl_rect)), std::back_inserter(cut_bg_rect_net_pair_list));
            }
            for (auto& [bg_env_rect, env_net_idx] : cut_bg_rect_net_pair_list) {
              PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(bg_env_rect);
              polygon_cut_list.push_back(cut_rect);
            }
          }
        }

        // 依次判断eol egde
        for (const int32_t& eol_idx : eol_edge_idx_set) {
          int32_t eol_edge_length = edge_length_list[eol_idx];
          // 基本条件 满足eol_width
          if (eol_edge_length >= max_eol_width) {
            continue;
          }
          PlanarRect eol_segment_max_rect  // segment所在的max rect
              = DRCUTIL.getBoundingBox({coord_list[getIdx(eol_idx - 1, coord_size)], coord_list[eol_idx], coord_list[getIdx(eol_idx + 1, coord_size)]});
          for (GTLRectInt& gtl_max_rect : polygon_gtl_rect_list) {
            if (DRCUTIL.isInside(DRCUTIL.convertToPlanarRect(gtl_max_rect), eol_segment_max_rect)) {
              eol_segment_max_rect = DRCUTIL.convertToPlanarRect(gtl_max_rect);
              break;
            }
          }
          // segment 所在的线rect
          PlanarRect eol_segment_rect = DRCUTIL.getBoundingBox({coord_list[getIdx(eol_idx - 1, coord_size)], coord_list[eol_idx]});
          // 拿到edge的方向
          Orientation orientation = DRCUTIL.getOrientation(coord_list[getIdx(eol_idx - 1, coord_size)], coord_list[eol_idx]);
          Direction direction = DRCUTIL.getDirection(coord_list[getIdx(eol_idx - 1, coord_size)], coord_list[eol_idx]);
          if (direction != Direction::kHorizontal && direction != Direction::kVertical) {
            DRCLOG.error(Loc::current(), "get wrong direction!");
          }
          PlanarRect eol_edge_rect = DRCUTIL.getBoundingBox({coord_list[eol_idx], coord_list[getIdx(eol_idx - 1, coord_size)]});
          PlanarRect pre_eol_coord_rect = DRCUTIL.getBoundingBox({coord_list[getIdx(eol_idx - 1, coord_size)]});
          PlanarRect eol_coord_rect = DRCUTIL.getBoundingBox({coord_list[eol_idx]});
          // 然后根据根据规则生成eol的查找区域，查找区域有两个：eol_spacing_rect 和 par_spacing_rect,par_spacing_rect 又分左边和右边
          std::vector<PlanarRect> eol_spacing_rect_list;
          std::vector<PlanarRect> ete_spacing_rect_list;
          std::vector<PlanarRect> par_spacing_left_rect_list;
          std::vector<PlanarRect> par_spacing_right_rect_list;
          PlanarRect max_check_rect;
          // 用来得到最大的check rect
          int32_t max_eol_within = 0;
          int32_t max_eol_spacing = 0;
          int32_t max_par_space = 0;
          int32_t max_par_within = 0;
          // 生成每个规则下的查找区域
          for (const EOLRule& eol_rule : eol_rule_list) {
            int32_t eol_spacing = eol_rule.eol_spacing;
            int32_t eol_width = eol_rule.eol_width;
            int32_t eol_within = eol_rule.eol_within;
            int32_t par_space = eol_rule.has_sub_trace_eol_width ? eol_rule.par_space - eol_edge_length : eol_rule.par_space;  // 这里需要注意sub eol条件
            int32_t par_within = eol_rule.par_within;
            // 更新max距离
            max_eol_within = std::max(max_eol_within, eol_within);
            max_eol_spacing = std::max(max_eol_spacing, eol_spacing);
            max_par_space = std::max(max_par_space, par_space);
            max_par_within = std::max(max_par_within, par_within);
            // 由于使用的是gtl_hole_polygon,所以默认方向是逆时针方向,基于当前segment进行膨胀
            if (rotation == Rotation::kCounterclockwise) {
              switch (orientation) {
                case Orientation::kWest:
                  eol_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_edge_rect, eol_within, 0, eol_within, eol_spacing));
                  par_spacing_left_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_coord_rect, par_space, par_within, 0, eol_within));
                  par_spacing_right_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_eol_coord_rect, 0, par_within, par_space, eol_within));
                  max_check_rect = DRCUTIL.getEnlargedRect(eol_edge_rect, std::max(max_eol_width, max_par_space), max_par_within,
                                                           std::max(max_eol_width, max_par_space), max_eol_spacing);
                  if (eol_rule.has_ete && eol_rule.ete_space != eol_spacing) {
                    ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_edge_rect, eol_within, 0, eol_within, eol_rule.ete_space));
                  } else {
                    ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_edge_rect, eol_within, 0, eol_within, eol_spacing));
                  }
                  break;
                case Orientation::kEast:
                  eol_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_edge_rect, eol_within, eol_spacing, eol_within, 0));
                  par_spacing_left_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_coord_rect, 0, eol_within, par_space, par_within));
                  par_spacing_right_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_eol_coord_rect, par_space, eol_within, 0, par_within));
                  max_check_rect = DRCUTIL.getEnlargedRect(eol_edge_rect, std::max(max_eol_width, max_par_space), max_eol_spacing,
                                                           std::max(max_eol_width, max_par_space), max_par_within);
                  if (eol_rule.has_ete && eol_rule.ete_space != eol_spacing) {
                    ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_edge_rect, eol_within, eol_rule.ete_space, eol_within, 0));
                  } else {
                    ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_edge_rect, eol_within, eol_spacing, eol_within, 0));
                  }
                  break;
                case Orientation::kSouth:
                  eol_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_edge_rect, eol_spacing, eol_within, 0, eol_within));
                  par_spacing_left_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_coord_rect, eol_within, par_space, par_within, 0));
                  par_spacing_right_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_eol_coord_rect, eol_within, 0, par_within, par_space));
                  max_check_rect = DRCUTIL.getEnlargedRect(eol_edge_rect, max_eol_spacing, std::max(max_eol_width, max_par_space), max_par_within,
                                                           std::max(max_eol_width, max_par_space));
                  if (eol_rule.has_ete && eol_rule.ete_space != eol_spacing) {
                    ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_edge_rect, eol_rule.ete_space, eol_within, 0, eol_within));
                  } else {
                    ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_edge_rect, eol_spacing, eol_within, 0, eol_within));
                  }
                  break;
                case Orientation::kNorth:
                  eol_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_edge_rect, 0, eol_within, eol_spacing, eol_within));
                  par_spacing_left_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_coord_rect, par_within, 0, eol_within, par_space));
                  par_spacing_right_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_eol_coord_rect, par_within, par_space, eol_within, 0));
                  max_check_rect = DRCUTIL.getEnlargedRect(eol_edge_rect, max_par_within, std::max(max_eol_width, max_par_space), max_eol_spacing,
                                                           std::max(max_eol_width, max_par_space));
                  if (eol_rule.has_ete && eol_rule.ete_space != eol_spacing) {
                    ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_edge_rect, 0, eol_within, eol_rule.ete_space, eol_within));
                  } else {
                    ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(eol_edge_rect, 0, eol_within, eol_spacing, eol_within));
                  }
                  break;
                default:
                  DRCLOG.error(Loc::current(), "The orientation is error!");
              }
            } else {
              DRCLOG.error(Loc::current(), "The rotation is error!");
            }
          }

          // 拿到周围所有的polygon
          std::vector<std::pair<BGRectInt, std::pair<int32_t, int32_t>>> bg_rect_net_pair_list;
          {
            GTLRectInt gtl_check_rect;
            routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(max_check_rect)),
                                                          std::back_inserter(bg_rect_net_pair_list));
          }

          std::map<int32_t, std::set<int32_t>> env_net_polygon_list_map;  // 存储周围的polygon,用net + idx来标记唯一的polygon
          for (auto& [bgrect, net_polygon_pair] : bg_rect_net_pair_list) {
            env_net_polygon_list_map[net_polygon_pair.first].insert(net_polygon_pair.second);
          }

          // 普通的left right par
          std::vector<bool> left_par(eol_rule_list.size(), false);
          std::vector<bool> right_par(eol_rule_list.size(), false);

          for (auto& [env_net_idx, env_poly_idx_set] : env_net_polygon_list_map) {
            for (const int32_t& env_poly_idx : env_poly_idx_set) {
              if (env_net_idx == net_idx && env_poly_idx == gtl_hole_poly_idx) {
                continue;  // 同一个polygon不参与par,这里存疑
              }
              GTLHolePolyInt& env_gtl_hole_poly = layer_net_polygon_list[routing_layer_idx][env_net_idx][env_poly_idx];
              // env_polygon_info
              int32_t env_coord_size = static_cast<int32_t>(env_gtl_hole_poly.size());
              if (env_coord_size < 4) {
                continue;
              }
              std::vector<Segment<PlanarCoord>> env_net_segment_list;
              std::vector<PlanarCoord> env_coord_list;
              std::vector<bool> env_convex_corner_list;
              std::vector<int32_t> env_edge_length_list;
              std::set<int32_t> env_eol_edge_idx_set;
              get_polygon_info(env_gtl_hole_poly, env_coord_size, env_coord_list, env_convex_corner_list, env_edge_length_list, env_net_segment_list,
                               env_eol_edge_idx_set);
              for (auto& env_segment : env_net_segment_list) {
                Direction env_direction = DRCUTIL.getDirection(env_segment.get_first(), env_segment.get_second());
                if (env_direction == direction) {
                  continue;  // par的segment必须与eol边垂直
                }
                PlanarRect env_segment_rect = DRCUTIL.getBoundingBox({env_segment.get_first(), env_segment.get_second()});
                for (int32_t eol_rule_idx = 0; eol_rule_idx < eol_rule_list.size(); eol_rule_idx++) {
                  // 不存在par规则的直接跳过 same metal后边再判断
                  if (eol_rule_list[eol_rule_idx].has_par == false || eol_rule_list[eol_rule_idx].has_same_metal) {
                    continue;
                  }
                  // 在left par rect
                  if (DRCUTIL.isOpenOverlap(env_segment_rect, par_spacing_left_rect_list[eol_rule_idx])) {
                    left_par[eol_rule_idx] = true;
                  }
                  // 在right par rect
                  if (DRCUTIL.isOpenOverlap(env_segment_rect, par_spacing_right_rect_list[eol_rule_idx])) {
                    right_par[eol_rule_idx] = true;
                  }
                }
              }
            }
          }

          for (auto& [env_net_idx, env_poly_idx_set] : env_net_polygon_list_map) {
            if (env_net_idx == -1 && net_idx == -1) {
              continue;
            }

            for (const int32_t& env_poly_idx : env_poly_idx_set) {
              GTLHolePolyInt& env_gtl_hole_poly = layer_net_polygon_list[routing_layer_idx][env_net_idx][env_poly_idx];
              // env_polygon_info
              int32_t env_coord_size = static_cast<int32_t>(env_gtl_hole_poly.size());
              if (env_coord_size < 4) {
                continue;
              }
              std::vector<Segment<PlanarCoord>> env_net_segment_list;
              std::vector<PlanarCoord> env_coord_list;
              std::vector<bool> env_convex_corner_list;
              std::vector<int32_t> env_edge_length_list;
              std::set<int32_t> env_eol_edge_idx_set;
              get_polygon_info(env_gtl_hole_poly, env_coord_size, env_coord_list, env_convex_corner_list, env_edge_length_list, env_net_segment_list,
                               env_eol_edge_idx_set);
              for (int32_t env_segment_idx = 0; env_segment_idx < env_net_segment_list.size(); env_segment_idx++) {
                Segment<PlanarCoord>& env_segment = env_net_segment_list[env_segment_idx];
                // eol 的另一条边满足direction相同，orirntation不同
                Direction env_direction = DRCUTIL.getDirection(env_segment.get_first(), env_segment.get_second());
                Orientation env_orientation = DRCUTIL.getOrientation(env_segment.get_first(), env_segment.get_second());
                if (!(env_direction == direction && env_orientation != orientation)) {
                  continue;
                }
                PlanarRect env_segment_rect = DRCUTIL.getBoundingBox({env_segment.get_first(), env_segment.get_second()});
                // 然后逐条规则进行匹配 从后往前
                for (int32_t eol_rule_idx = eol_rule_list.size() - 1; eol_rule_idx >= 0; eol_rule_idx--) {
                  EOLRule cur_rule = eol_rule_list[eol_rule_idx];
                  if (eol_edge_length >= cur_rule.eol_width) {
                    continue;  // 不满足eol width
                  }
                  // 如果有min length判断min length
                  if (cur_rule.has_min_length) {
                    int32_t min_length = cur_rule.min_length;
                    if (edge_length_list[getIdx(eol_idx - 1, coord_size)] < min_length && edge_length_list[getIdx(eol_idx + 1, coord_size)] < min_length) {
                      continue;  // 不满足min length
                    }
                  }

                  // 如果有cut相关的判断cut的规则 cut不满足直接跳出
                  if (cur_rule.has_enclose_cut) {
                    int32_t enclosed_dist = cur_rule.enclosed_dist;
                    int32_t cut_to_metal_space = cur_rule.cut_to_metal_space;
                    // 拿到当前矩形下cut
                    bool is_cut_require = false;
                    // ALL CUT要求所有的都满足，但是实测下来并非如此
                    for (PlanarRect& cut_rect : polygon_cut_list) {
                      if (DRCUTIL.isClosedOverlap(eol_segment_max_rect, cut_rect)) {
                        int32_t cut_to_segment_dis = DRCUTIL.getEuclideanDistance(cut_rect, eol_segment_rect);
                        int32_t cut_to_env_dis = DRCUTIL.getEuclideanDistance(cut_rect, env_segment_rect);
                        if (cut_to_segment_dis < enclosed_dist && cut_to_env_dis < cut_to_metal_space) {
                          is_cut_require = true;
                          break;
                        }
                      }
                    }
                    if (is_cut_require == false) {
                      continue;
                    }
                  }
                  // 如果有par判断par
                  if (cur_rule.has_par) {
                    // same metal的单独判断
                    if (cur_rule.has_same_metal) {
                      int32_t par_within = cur_rule.par_within;
                      Segment<PlanarCoord> pre_env_segment = env_net_segment_list[getIdx(env_segment_idx - 1, env_net_segment_list.size())];
                      Segment<PlanarCoord> post_env_segment = env_net_segment_list[getIdx(env_segment_idx + 1, env_net_segment_list.size())];
                      PlanarRect pre_env_segment_rect = DRCUTIL.getBoundingBox({pre_env_segment.get_first(), pre_env_segment.get_second()});
                      PlanarRect post_env_segment_rect = DRCUTIL.getBoundingBox({post_env_segment.get_first(), post_env_segment.get_second()});
                      bool pre_left = false;
                      bool pre_right = false;
                      if (DRCUTIL.isOpenOverlap(pre_env_segment_rect, par_spacing_left_rect_list[eol_rule_idx])
                          && DRCUTIL.getParallelLength(pre_env_segment_rect, par_spacing_left_rect_list[eol_rule_idx]) < par_within) {
                        pre_left = true;
                      } else if (DRCUTIL.isOpenOverlap(pre_env_segment_rect, par_spacing_right_rect_list[eol_rule_idx])
                                 && DRCUTIL.getParallelLength(pre_env_segment_rect, par_spacing_right_rect_list[eol_rule_idx]) < par_within) {
                        pre_right = true;
                      }

                      bool post_left = false;
                      bool post_right = false;
                      if (DRCUTIL.isOpenOverlap(post_env_segment_rect, par_spacing_left_rect_list[eol_rule_idx])
                          && DRCUTIL.getParallelLength(post_env_segment_rect, par_spacing_left_rect_list[eol_rule_idx]) < par_within) {
                        pre_left = true;
                      } else if (DRCUTIL.isOpenOverlap(post_env_segment_rect, par_spacing_right_rect_list[eol_rule_idx])
                                 && DRCUTIL.getParallelLength(post_env_segment_rect, par_spacing_right_rect_list[eol_rule_idx]) < par_within) {
                        pre_right = true;
                      }
                      if (cur_rule.has_two_edges) {
                        if (!(pre_left && post_right) && !(pre_right && post_left)) {
                          continue;  // 两边都需要满足
                        }
                      } else {
                        if (!(pre_left || pre_right) && !(post_left || post_right)) {
                          continue;  // 至少一边满足
                        }
                      }
                    } else {
                      if (cur_rule.has_two_edges) {
                        if (!(left_par[eol_rule_idx] && right_par[eol_rule_idx])) {
                          continue;  // 两边都需要满足
                        }
                      } else {
                        if (!(left_par[eol_rule_idx] || right_par[eol_rule_idx])) {
                          continue;  // 至少一边满足
                        }
                      }
                    }
                  }

                  // 如果有ete那么判断ete
                  PlanarRect eol_spacing_rect = eol_spacing_rect_list[eol_rule_idx];
                  if (cur_rule.has_ete) {
                    // 判断是否为ete
                    if (DRCUTIL.exist(env_eol_edge_idx_set, env_segment_idx) && env_edge_length_list[env_segment_idx] < cur_rule.eol_width) {
                      eol_spacing_rect = ete_spacing_rect_list[eol_rule_idx];
                    }
                  }
                  // 满足当前所有规则那么生成违例矩形并break
                  PlanarRect violation_rect = DRCUTIL.getSpacingRect(eol_segment_rect, env_segment_rect);
                  int32_t require_size = direction == Direction::kHorizontal ? eol_spacing_rect.getYSpan() : eol_spacing_rect.getXSpan();
                  if (!DRCUTIL.isOpenOverlap(eol_spacing_rect, env_segment_rect)) {
                    break;  // 在这一级的距离不满足那么前一级的距离更不满足
                  }
                  Violation violation;
                  violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                  violation.set_required_size(require_size);
                  violation.set_is_routing(true);
                  violation.set_violation_net_set({net_idx, env_net_idx});
                  violation.set_layer_idx(routing_layer_idx);
                  violation.set_rect(violation_rect);
                  // rv_box.get_violation_list().push_back(violation);
                  edge_vio_map[edge_list[eol_idx]].push_back(std::make_pair(env_net_segment_list[env_segment_idx], violation));
                  break;  // 满足这一级，直接break
                }
              }
            }
          }
          // eol edge
        }
        // polygon
      }
    }

    // 每个segment保留最短距离的违例
    std::set<Violation, CmpViolation> no_need_vio;
    for (auto& [segment, segment_vio_list] : edge_vio_map) {
      bool use_x_span = segment.get_first().get_x() == segment.get_second().get_x();
      int32_t best_idx = 0;

      int32_t best_length = use_x_span ? segment_vio_list[0].second.getXSpan() : segment_vio_list[0].second.getYSpan();
      for (int32_t i = 0; i < segment_vio_list.size(); i++) {
        int32_t now_length = use_x_span ? segment_vio_list[i].second.getXSpan() : segment_vio_list[i].second.getYSpan();
        if (now_length < best_length) {
          best_length = now_length;
          best_idx = i;
        }
      }
      for (int32_t i = 0; i < segment_vio_list.size(); i++) {
        bool use_x_span = segment.get_first().get_x() == segment.get_second().get_x();
        int32_t now_length = use_x_span ? segment_vio_list[i].second.getXSpan() : segment_vio_list[i].second.getYSpan();
        if (best_length == now_length) {
          continue;
        }
        no_need_vio.insert(segment_vio_list[i].second);
      }
    }

    for (auto& [segment, segment_vio_list] : edge_vio_map) {
      for (auto& [env_segment, violation] : segment_vio_list) {
        if (DRCUTIL.exist(no_need_vio, violation)) {
          continue;
        }
        rv_box.get_violation_list().push_back(violation);
      }
    }
  }
}
}  // namespace idrc
