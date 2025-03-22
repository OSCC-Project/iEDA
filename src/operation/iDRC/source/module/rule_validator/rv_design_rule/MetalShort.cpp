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

// anonymous namespace
namespace {
// using Rtree for query net id in violation rect
using BgBoxInt = boost::geometry::model::box<BGPointInt>;
using RTree = bgi::rtree<std::pair<BgBoxInt, int32_t>, bgi::quadratic<16>>;
using LayerRTreeMap = std::map<int32_t, RTree>;

void addRectToRtree(LayerRTreeMap& _query_tree, GTLRectInt rect, int32_t layer_idx, int32_t net_idx)
{
  BgBoxInt rtree_rect(BGPointInt(xl(rect), yl(rect)), BGPointInt(xh(rect), yh(rect)));
  _query_tree[layer_idx].insert(std::make_pair(rtree_rect, net_idx));
}

std::set<int32_t> queryNetIdbyRtree(LayerRTreeMap& _query_tree, int32_t layer_idx, int32_t llx, int32_t lly, int32_t urx, int32_t ury)
{
  std::set<int32_t> net_ids;
  std::vector<std::pair<BgBoxInt, int32_t>> result;
  BgBoxInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
  _query_tree[layer_idx].query(bgi::intersects(rect), std::back_inserter(result));
  for (auto& pair : result) {
    net_ids.insert(pair.second);
  }
  return net_ids;
}

auto shrink_rect = [](GTLRectInt& rect, int value) -> bool {
  int width = gtl::delta(rect, gtl::HORIZONTAL);
  int height = gtl::delta(rect, gtl::VERTICAL);
  if (width < 2 * value || height < 2 * value) {
    return false;
  }

  gtl::shrink(rect, gtl::HORIZONTAL, value);
  gtl::shrink(rect, gtl::VERTICAL, value);

  return true;
};

GTLPolySetInt& get_no_overlap_poly_set(GTLPolySetInt& poly_set)
{
  GTLPolySetInt poly_set_copy = poly_set;
  poly_set_copy.self_intersect();
  poly_set -= poly_set_copy;
  return poly_set;
}
}  // namespace

namespace idrc {

void RuleValidator::verifyMetalShort(RVBox& rv_box)
{
  LayerRTreeMap layer_query_tree;

  std::vector<DRCShape*>& drc_env_shape_list = rv_box.get_drc_env_shape_list();
  std::vector<DRCShape*>& drc_result_shape_list = rv_box.get_drc_result_shape_list();
  std::vector<Violation>& violation_list = rv_box.get_violation_list();

  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> layer_net_gtl_poly_set_map;
  std::map<int32_t, GTLPolySetInt> layer_gtl_poly_set_map;

  for (DRCShape* drc_shape : drc_env_shape_list) {
    if (!drc_shape->get_is_routing()) {
      continue;
    }
    int32_t layer_idx = drc_shape->get_layer_idx();
    int32_t net_idx = drc_shape->get_net_idx();

    layer_net_gtl_poly_set_map[layer_idx][net_idx] += GTLRectInt(drc_shape->get_ll_x(), drc_shape->get_ll_y(), 
                                                               drc_shape->get_ur_x(), drc_shape->get_ur_y());
  }

  for (DRCShape* drc_shape : drc_result_shape_list) {
    if (!drc_shape->get_is_routing() || drc_shape->get_net_idx() == -1) {
      continue;
    }
    int32_t layer_idx = drc_shape->get_layer_idx();
    int32_t net_idx = drc_shape->get_net_idx();

    layer_net_gtl_poly_set_map[layer_idx][net_idx] += GTLRectInt(drc_shape->get_ll_x(), drc_shape->get_ll_y(), 
                                                               drc_shape->get_ur_x(), drc_shape->get_ur_y());
  }

  for (auto& [layer_idx, net_gtl_poly_set_map] : layer_net_gtl_poly_set_map) {
    for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
      gtl_poly_set.clean();

      std::vector<GTLHolePolyInt> gtl_hole_poly_list;
      gtl_poly_set.get(gtl_hole_poly_list);
      for (auto& gtl_hole_poly : gtl_hole_poly_list) {
        GTLRectInt query_rect;
        gtl::extents(query_rect, gtl_hole_poly);
        addRectToRtree(layer_query_tree, query_rect, layer_idx, net_idx);
      }
      gtl_poly_set.bloat2(1,1,1,1);
      layer_gtl_poly_set_map[layer_idx] += gtl_poly_set;
    }
  }
  
  for (auto& [layer_idx, gtl_poly_set] : layer_gtl_poly_set_map) {
    std::vector<gtl::polygon_90_with_holes_data<int32_t>> overlap_list;
    auto& intersets = gtl_poly_set.self_intersect();
    intersets.get(overlap_list);
    for (auto& overlap_poly : overlap_list) {
      std::vector<GTLRectInt> rect_results;
      gtl::get_max_rectangles(rect_results, overlap_poly);
      for (auto gtl_rect : rect_results) {
        if(!shrink_rect(gtl_rect, 1)){
          continue;
        }
        int width = gtl::delta(gtl_rect, gtl::HORIZONTAL);
        int height = gtl::delta(gtl_rect, gtl::VERTICAL);
        if (width == 0 || height == 0) {
          continue;
        }
        int llx = gtl::xl(gtl_rect);
        int lly = gtl::yl(gtl_rect);
        int urx = gtl::xh(gtl_rect);
        int ury = gtl::yh(gtl_rect);

        std::set<int32_t> net_set = queryNetIdbyRtree(layer_query_tree, layer_idx, llx, lly, urx, ury);

        if (net_set.empty() || net_set.size() == net_set.count(-1)) {
          continue;
        }

        // Create violations for each pair if more than two nets
        if (net_set.size() > 2) {
          std::vector<int32_t> net_vec(net_set.begin(), net_set.end());

          for (size_t i = 0; i < net_vec.size(); ++i) {
            for (size_t j = i + 1; j < net_vec.size(); ++j) {
              std::set<int32_t> pair_net_set;
              pair_net_set.insert(net_vec[i]);
              pair_net_set.insert(net_vec[j]);

              Violation violation;
              violation.set_violation_type(ViolationType::kMetalShort);
              violation.set_required_size(0);
              violation.set_is_routing(true);
              violation.set_violation_net_set(pair_net_set);
              violation.set_layer_idx(layer_idx);
              violation.set_rect(PlanarRect(llx, lly, urx, ury));

              violation_list.push_back(violation);
            }
          }
        } else {
          Violation violation;
          violation.set_violation_type(ViolationType::kMetalShort);
          violation.set_required_size(0);
          violation.set_is_routing(true);
          violation.set_violation_net_set(net_set);
          violation.set_layer_idx(layer_idx);
          violation.set_rect(PlanarRect(llx, lly, urx, ury));

          violation_list.push_back(violation);
        }
      }
    }
  }
}
}  // namespace idrc
