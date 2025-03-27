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
}  // namespace
namespace idrc {

using GTLSegment = gtl::segment_data<int32_t>;
// 记录notch spacing的信息
struct NotchSpaingRule
{
  int32_t notch_spacing;
  int32_t notch_length;
  int32_t concave_ends;
  bool has_concave_ends;
};

std::map<int32_t, NotchSpaingRule> layer_notch_spacing_rule = {
    {0, {70 * 2, 155 * 2, 55 * 2, true}},  // M1
    {1, {70 * 2, 155 * 2, 55 * 2, true}},  // M2
    {2, {70 * 2, 155 * 2, 0, false}},      // M3
    {3, {70 * 2, 155 * 2, 0, false}},      // M4
    {4, {70 * 2, 155 * 2, 0, false}},      // M5
    {5, {70 * 2, 155 * 2, 0, false}},      // M6
    {6, {70 * 2, 155 * 2, 0, false}}       // M7
};

// 记录point的信息
struct StepPoint
{
  GTLPointInt point;
  bool is_convex = true;
};

int32_t get_cross_func(const GTLPointInt& left, const GTLPointInt& middle, const GTLPointInt& right)
{
  int32_t x1 = gtl::x(left) - gtl::x(middle);
  int32_t y1 = gtl::y(left) - gtl::y(middle);
  int32_t x2 = gtl::x(right) - gtl::x(middle);
  int32_t y2 = gtl::y(right) - gtl::y(middle);
  return x1 * y2 - x2 * y1;
}

void get_point_and_egde_by_hole_poly(GTLHolePolyInt& hole_poly, std::vector<StepPoint>& step_points, std::vector<GTLSegment>& step_edges)
{
  // 得到所有点
  for (GTLPointInt point : hole_poly) {
    StepPoint temp_point;
    temp_point.point = point;
    step_points.push_back(temp_point);
  }
  // 得到所有的点的凹凸性并得到所有的边
  for (int32_t i = 0; i < step_points.size(); i++) {
    StepPoint& middle = step_points[i];
    StepPoint& left = step_points[(i - 1 + step_points.size()) % step_points.size()];
    StepPoint& right = step_points[(i + 1) % step_points.size()];
    int32_t cross_product = get_cross_func(left.point, middle.point, right.point);
    // boost采用逆时针存储的点，叉积大于0为凹点
    if (cross_product < 0) {
      middle.is_convex = true;
    } else if (cross_product > 0) {
      middle.is_convex = false;
    } else {
      DRCLOG.error(Loc::current(), "NotchSpacing: three points are in a line!");
    }
    // 得到所有的边
    step_edges.push_back(GTLSegment(left.point, middle.point));
  }
}
void get_point_and_egde_by_hole(GTLPolyInt& hole, std::vector<StepPoint>& step_points, std::vector<GTLSegment>& step_edges)
{
  // 得到所有点
  for (GTLPointInt point : hole) {
    StepPoint temp_point;
    temp_point.point = point;
    step_points.push_back(temp_point);
  }
  // 得到所有的点的凹凸性并得到所有的边
  for (int32_t i = 0; i < step_points.size(); i++) {
    StepPoint& middle = step_points[i];
    StepPoint& left = step_points[(i - 1 + step_points.size()) % step_points.size()];
    StepPoint& right = step_points[(i + 1) % step_points.size()];
    int32_t cross_product = get_cross_func(left.point, middle.point, right.point);
    // boost采用逆时针存储的点，叉积大于0为凹点,因为是hole，所以要反过来,但是遍历顺序不一样，所以不用反
    if (cross_product < 0) {
      middle.is_convex = true;
    } else if (cross_product > 0) {
      middle.is_convex = false;
    } else {
      DRCLOG.error(Loc::current(), "NotchSpacing: three points are in a line!");
    }
    // 得到所有的边
    step_edges.push_back(GTLSegment(left.point, middle.point));
  }
}

void convex_notch_check(GTLSegment& concave_edge, GTLSegment& spacing_edge, GTLSegment& side_edge, GTLHolePolyInt& polygon, int32_t layer_idx,
                        std::map<int32_t, GTLPolySetInt>& layer_violation_poly_set)
{
  int32_t notch_spacing = layer_notch_spacing_rule[layer_idx].notch_spacing;
  int32_t notch_length = layer_notch_spacing_rule[layer_idx].notch_length;
  int32_t concave_ends = layer_notch_spacing_rule[layer_idx].concave_ends;

  int32_t concave_edge_length = gtl::length(concave_edge);
  int32_t side_edge_length = gtl::length(side_edge);
  int32_t spacing_edge_length = gtl::length(spacing_edge);
  if (concave_edge_length < notch_length && side_edge_length >= notch_length
      && spacing_edge_length < notch_spacing) {  // 满足这三个条件，然后判断两边的矩形是否满足条件
    gtl::orientation_2d_enum slice_dir = gtl::x(spacing_edge.low()) == gtl::x(spacing_edge.high()) ? gtl::HORIZONTAL : gtl::VERTICAL;  // 水平竖切，竖直横切
    // 找到两条边对应的两个矩形，对于竖直凹槽，左边为矩形右边，右边为矩形左边，对于水平凹槽，上边为矩形下边，下边为矩形上边
    std::vector<GTLRectInt> slice_rect_list;
    // gtl::get_rectangles(slice_rect_list, polygon, slice_dir);
    gtl::get_max_rectangles(slice_rect_list, polygon);
    if (slice_dir == gtl::HORIZONTAL) {
      GTLSegment& low_edge = gtl::y(concave_edge.low()) < gtl::y(side_edge.low()) ? concave_edge : side_edge;
      GTLSegment& high_edge = gtl::y(concave_edge.low()) > gtl::y(side_edge.low()) ? concave_edge : side_edge;
      bool find_low = false;
      bool find_high = false;

      for (GTLRectInt& slice_rect : slice_rect_list) {
        if (find_low && find_high) {
          break;
        }
        GTLSegment slice_rect_low_edge
            = GTLSegment(GTLPointInt(gtl::xl(slice_rect), gtl::yl(slice_rect)), GTLPointInt(gtl::xh(slice_rect), gtl::yl(slice_rect)));
        GTLSegment slice_rect_high_edge
            = GTLSegment(GTLPointInt(gtl::xl(slice_rect), gtl::yh(slice_rect)), GTLPointInt(gtl::xh(slice_rect), gtl::yh(slice_rect)));
        if (gtl::contains(slice_rect_low_edge, high_edge)) {
          int32_t rect_width = gtl::yh(slice_rect) - gtl::yl(slice_rect);
          if (rect_width > concave_ends) {
            find_high = false;
            break;
          } else {
            find_high = true;
          }
        }
        if (gtl::contains(slice_rect_high_edge, low_edge)) {
          int32_t rect_width = gtl::yh(slice_rect) - gtl::yl(slice_rect);
          if (rect_width > concave_ends) {
            find_low = false;
            break;
          } else {
            find_low = true;
          }
        }
      }
      if (find_low && find_high) {
        // 画出违例区域
        GTLSegment& violation_rect_first = spacing_edge;
        GTLSegment& violation_rect_second = gtl::length(concave_edge) < gtl::length(side_edge) ? concave_edge : side_edge;
        GTLPointInt point1 = violation_rect_first.low();
        GTLPointInt point2 = violation_rect_first.high();
        GTLPointInt point3 = violation_rect_second.low();
        GTLPointInt point4 = violation_rect_second.high();
        int32_t llx = std::min(std::min(gtl::x(point1), gtl::x(point2)), std::min(gtl::x(point3), gtl::x(point4)));
        int32_t lly = std::min(std::min(gtl::y(point1), gtl::y(point2)), std::min(gtl::y(point3), gtl::y(point4)));
        int32_t urx = std::max(std::max(gtl::x(point1), gtl::x(point2)), std::max(gtl::x(point3), gtl::x(point4)));
        int32_t ury = std::max(std::max(gtl::y(point1), gtl::y(point2)), std::max(gtl::y(point3), gtl::y(point4)));

        layer_violation_poly_set[layer_idx] += (GTLRectInt(llx, lly, urx, ury));
      }
    } else {
      GTLSegment& low_edge = gtl::x(concave_edge.low()) < gtl::x(side_edge.low()) ? concave_edge : side_edge;
      GTLSegment& high_edge = gtl::x(concave_edge.low()) > gtl::x(side_edge.low()) ? concave_edge : side_edge;
      bool find_low = false;
      bool find_high = false;
      for (GTLRectInt& slice_rect : slice_rect_list) {
        if (find_low && find_high) {
          break;
        }
        GTLSegment slice_rect_low_edge
            = GTLSegment(GTLPointInt(gtl::xl(slice_rect), gtl::yl(slice_rect)), GTLPointInt(gtl::xl(slice_rect), gtl::yh(slice_rect)));
        GTLSegment slice_rect_high_edge
            = GTLSegment(GTLPointInt(gtl::xh(slice_rect), gtl::yl(slice_rect)), GTLPointInt(gtl::xh(slice_rect), gtl::yh(slice_rect)));
        if (gtl::contains(slice_rect_low_edge, high_edge)) {
          int32_t rect_width = gtl::xh(slice_rect) - gtl::xl(slice_rect);
          if (rect_width > concave_ends) {
            find_high = false;
            break;
          } else {
            find_high = true;
          }
        }
        if (gtl::contains(slice_rect_high_edge, low_edge)) {
          int32_t rect_width = gtl::xh(slice_rect) - gtl::xl(slice_rect);
          if (rect_width > concave_ends) {
            find_low = false;
            break;
          } else {
            find_low = true;
          }
        }
      }
      if (find_low && find_high) {
        // 画出违例区域
        GTLSegment& violation_rect_first = spacing_edge;
        GTLSegment& violation_rect_second = gtl::length(concave_edge) < gtl::length(side_edge) ? concave_edge : side_edge;
        GTLPointInt point1 = violation_rect_first.low();
        GTLPointInt point2 = violation_rect_first.high();
        GTLPointInt point3 = violation_rect_second.low();
        GTLPointInt point4 = violation_rect_second.high();
        int32_t llx = std::min(std::min(gtl::x(point1), gtl::x(point2)), std::min(gtl::x(point3), gtl::x(point4)));
        int32_t lly = std::min(std::min(gtl::y(point1), gtl::y(point2)), std::min(gtl::y(point3), gtl::y(point4)));
        int32_t urx = std::max(std::max(gtl::x(point1), gtl::x(point2)), std::max(gtl::x(point3), gtl::x(point4)));
        int32_t ury = std::max(std::max(gtl::y(point1), gtl::y(point2)), std::max(gtl::y(point3), gtl::y(point4)));

        layer_violation_poly_set[layer_idx] += (GTLRectInt(llx, lly, urx, ury));
      }
    }
    // 水平凹槽
  }
}
void check_by_point_and_edge(std::vector<StepPoint>& step_points, std::vector<GTLSegment>& step_edges, GTLHolePolyInt& hole_poly, int32_t layer_idx,
                             std::map<int32_t, GTLPolySetInt>& layer_violation_poly_set)
{
  int32_t notch_spacing = layer_notch_spacing_rule[layer_idx].notch_spacing;
  int32_t notch_length = layer_notch_spacing_rule[layer_idx].notch_length;
  int32_t concave_ends = layer_notch_spacing_rule[layer_idx].concave_ends;
  bool has_concave_ends = layer_notch_spacing_rule[layer_idx].has_concave_ends;
  for (int32_t i = 0; i < step_points.size(); i++) {
    //DEBUG
    // if(step_points[i].point == GTLPointInt(38450,685450)){
    //   int debug = 0;
    // }
    if (step_points[i].is_convex) {
      continue;  // 非凹点，跳过
    }
    int32_t before_index = (i - 1 + step_points.size()) % step_points.size();
    int32_t second_index = (i + 1) % step_points.size();
    int32_t third_index = (i + 2) % step_points.size();
    int32_t fourth_index = (i + 3) % step_points.size();
    if (has_concave_ends == false) {  // 只检查两个凹角
      if (step_points[second_index].is_convex) {
        continue;
      }
      GTLSegment& side_edge_a = step_edges[i];
      GTLSegment& spacing_edge = step_edges[second_index];
      GTLSegment& side_edge_b = step_edges[third_index];

      // 第一条边或第三条边应该小于notch_length,第二条边的长度应该小于notch_spacing
      int32_t length_a = gtl::length(side_edge_a);
      int32_t length_b = gtl::length(side_edge_b);
      int32_t length_spacing = gtl::length(spacing_edge);
      if ((length_a < notch_length || length_b < notch_length) && (length_spacing < notch_spacing)) {
        // 用第二条边和短的那条边构成违例区域
        GTLSegment violation_rect_first = spacing_edge;
        GTLSegment violation_rect_second = length_a < length_b ? side_edge_a : side_edge_b;

        GTLPointInt point1 = violation_rect_first.low();
        GTLPointInt point2 = violation_rect_first.high();
        GTLPointInt point3 = violation_rect_second.low();
        GTLPointInt point4 = violation_rect_second.high();
        int32_t llx = std::min(std::min(gtl::x(point1), gtl::x(point2)), std::min(gtl::x(point3), gtl::x(point4)));
        int32_t lly = std::min(std::min(gtl::y(point1), gtl::y(point2)), std::min(gtl::y(point3), gtl::y(point4)));
        int32_t urx = std::max(std::max(gtl::x(point1), gtl::x(point2)), std::max(gtl::x(point3), gtl::x(point4)));
        int32_t ury = std::max(std::max(gtl::y(point1), gtl::y(point2)), std::max(gtl::y(point3), gtl::y(point4)));

        layer_violation_poly_set[layer_idx] += GTLRectInt(llx, lly, urx, ury);
      }
    } else {  // 多个凹角的情况 // 检查三凹角三个连续的凹角，根据lef的示意图，四凹角的先不管？
      if (step_points[second_index].is_convex || step_points[third_index].is_convex) {
        continue;
      }
      // 这时是四凹角，先不用管这种情况
      if (!step_points[fourth_index].is_convex || !step_points[before_index].is_convex) {
        continue;
      }
      // 三凹角会有四条边，对应两种情况：1 2作为底边，2 3作为底边
      /*
     1-----
     |
     |          |
     2----------3
      */
      GTLSegment& first_edge = step_edges[i];
      GTLSegment& second_edge = step_edges[second_index];
      GTLSegment& third_edge = step_edges[third_index];
      GTLSegment& fourth_edge = step_edges[fourth_index];
      convex_notch_check(third_edge, second_edge, first_edge, hole_poly, layer_idx, layer_violation_poly_set);
      convex_notch_check(second_edge, third_edge, fourth_edge, hole_poly, layer_idx, layer_violation_poly_set);
    }
  }
}
void RuleValidator::verifyNotchSpacing(RVBox& rv_box)
{
  // return;
  /*
    t28中的notch spacing(110 没有该规则,110下直接跳过该规则的检查):
    PROPERTY LEF58_SPACING "
    SPACING 0.07 NOTCHLENGTH 0.155 CONCAVEENDS 0.055 ; " ;
   */
  std::vector<Violation>& violation_list = rv_box.get_violation_list();
  /*R-tree新写的逻辑*/
  LayerRTreeMap layer_query_tree;
  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> layer_net_poly_set;
  std::map<int32_t, GTLPolySetInt> layer_violation_poly_set;
  for (DRCShape* rect : rv_box.get_drc_result_shape_list()) {
    if (!rect->get_is_routing() && rect->get_net_idx() == -1) {  // 不是routing layer或者net_idx为-1的跳过该检测
      continue;
    }
    int32_t layer_idx = rect->get_layer_idx();
    int32_t net_idx = rect->get_net_idx();
    addRectToRtree(layer_query_tree, GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y()), layer_idx, net_idx);
    layer_net_poly_set[layer_idx][net_idx] += GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
  }
  for (DRCShape* rect : rv_box.get_drc_env_shape_list()) {
    if (!rect->get_is_routing() && rect->get_net_idx() == -1) {  // 不是routing layer或者net_idx为-1的跳过该检测
      continue;
    }
    int32_t layer_idx = rect->get_layer_idx();
    int32_t net_idx = rect->get_net_idx();
    addRectToRtree(layer_query_tree, GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y()), layer_idx, net_idx);
    layer_net_poly_set[layer_idx][net_idx] += GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
  }

  for (auto& [layer_idx, net_poly_set] : layer_net_poly_set) {
    for (auto& [net_idx, poly_set] : net_poly_set) {
      std::vector<GTLHolePolyInt> hole_poly_list;
      poly_set.get(hole_poly_list);  // get会自动识别要变成的类型,用带hole能够避免识别到hole导致edge不对
      for (GTLHolePolyInt& hole_poly : hole_poly_list) {
        std::vector<StepPoint> step_points;
        std::vector<GTLSegment> step_edges;
        get_point_and_egde_by_hole_poly(hole_poly, step_points, step_edges);
        check_by_point_and_edge(step_points, step_edges, hole_poly, layer_idx, layer_violation_poly_set);
        // hole也要判断
        GTLHolePolyInt::iterator_holes_type hole_iter = hole_poly.begin_holes();
        while (hole_iter != hole_poly.end_holes()) {
          GTLPolyInt hole = *hole_iter;
          std::vector<StepPoint> step_points;
          std::vector<GTLSegment> step_edges;
          get_point_and_egde_by_hole(hole, step_points, step_edges);
          // 找到连续的凹点，进而判断notch spacing
          check_by_point_and_edge(step_points, step_edges, hole_poly, layer_idx, layer_violation_poly_set);

          hole_iter++;
        }
      }
    }
  }

  for (auto& [layer_idx, violation_poly_set] : layer_violation_poly_set) {
    std::vector<GTLRectInt> violation_rect_list;
    gtl::get_max_rectangles(violation_rect_list, violation_poly_set);
    int32_t required_size = layer_notch_spacing_rule[layer_idx].notch_spacing;
    for (GTLRectInt& violation_rect : violation_rect_list) {
      int32_t llx = gtl::xl(violation_rect);
      int32_t lly = gtl::yl(violation_rect);
      int32_t urx = gtl::xh(violation_rect);
      int32_t ury = gtl::yh(violation_rect);

      std::set<int32_t> net_set = queryNetIdbyRtree(layer_query_tree, layer_idx, llx, lly, urx, ury);
      Violation violation;
      violation.set_violation_type(ViolationType::kNotchSpacing);
      violation.set_required_size(required_size);
      violation.set_is_routing(true);
      violation.set_violation_net_set(net_set);
      violation.set_layer_idx(layer_idx);
      violation.set_rect(PlanarRect(llx, lly, urx, ury));

      violation_list.push_back(violation);
    }
  }
}

}  // namespace idrc
