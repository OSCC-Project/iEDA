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
}  // namespace

namespace idrc {

void RuleValidator::verifyCutShort(RVBox& rv_box)
{
  LayerRTreeMap layer_query_tree;
  std::map<int32_t, GTLPolySetInt> total_layer_poly_set;

  std::vector<DRCShape*>& drc_env_shape_list = rv_box.get_drc_env_shape_list();
  std::vector<DRCShape*>& drc_result_shape_list = rv_box.get_drc_result_shape_list();
  std::vector<Violation>& violation_list = rv_box.get_violation_list();

  for (DRCShape* shape : drc_result_shape_list) {
    if (shape->get_is_routing() == true) {
      continue;
    }
    int32_t layer_idx = shape->get_layer_idx();
    int32_t net_idx = shape->get_net_idx();
    if (net_idx == -1) {
      continue;
    }
    int32_t llx = shape->get_ll_x();
    int32_t lly = shape->get_ll_y();
    int32_t urx = shape->get_ur_x();
    int32_t ury = shape->get_ur_y();
    // expand rect by 1 unit,used to check interaction
    GTLRectInt rect(llx - 1, lly - 1, urx + 1, ury + 1);
    total_layer_poly_set[layer_idx] += rect;
    addRectToRtree(layer_query_tree, GTLRectInt(llx, lly, urx, ury), layer_idx, net_idx);
  }

  for (DRCShape* shape : drc_env_shape_list) {
    if (shape->get_is_routing() == true) {
      continue;
    }
    int32_t layer_idx = shape->get_layer_idx();
    int32_t net_idx = shape->get_net_idx();
    // if(net_idx == -1) {
    //   continue;
    // }
    int32_t llx = shape->get_ll_x();
    int32_t lly = shape->get_ll_y();
    int32_t urx = shape->get_ur_x();
    int32_t ury = shape->get_ur_y();
    // expand rect by 1 unit,used to check interaction
    GTLRectInt rect(llx - 1, lly - 1, urx + 1, ury + 1);
    total_layer_poly_set[layer_idx] += rect;
    addRectToRtree(layer_query_tree, GTLRectInt(llx, lly, urx, ury), layer_idx, net_idx);
  }

  for (auto& [layer_idx, poly_set] : total_layer_poly_set) {
    std::vector<gtl::polygon_90_with_holes_data<int32_t>> overlap_list;
    poly_set.self_intersect();
    poly_set.get(overlap_list);
    for (auto& overlap : overlap_list) {
      std::vector<GTLRectInt> results;
      gtl::get_max_rectangles(results, overlap);
      for (auto rect : results) {
        if (!shrink_rect(rect, 1)) {
          continue;
        }
        int llx = gtl::xl(rect);
        int lly = gtl::yl(rect);
        int urx = gtl::xh(rect);
        int ury = gtl::yh(rect);

        std::set<int32_t> net_set = queryNetIdbyRtree(layer_query_tree, layer_idx, llx, lly, urx, ury);

        // set cut short violation
        Violation violation;
        violation.set_violation_type(ViolationType::kCutShort);
        violation.set_required_size(0);
        violation.set_is_routing(true);
        violation.set_violation_net_set(net_set);
        violation.set_layer_idx(layer_idx - 1);
        violation.set_rect(PlanarRect(llx, lly, urx, ury));

        violation_list.push_back(violation);
      }
    }
  }
  // DRCLOG.info(Loc::current(), "cut short num: ", violation_list.size());
}

}  // namespace idrc
