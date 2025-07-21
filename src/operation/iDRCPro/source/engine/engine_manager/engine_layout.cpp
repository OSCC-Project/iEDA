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
#include "engine_layout.h"

#include <cstdio>

#include "engine_geometry_creator.h"
#include "geometry_boost.h"
#include "idrc_dm.h"

namespace idrc {
DrcEngineLayout::DrcEngineLayout(std::string layer) : _layer(layer)
{
  _layer = layer;
  ieda_solver::EngineGeometryCreator geo_creator;
  _engine = geo_creator.create();
}

DrcEngineLayout::~DrcEngineLayout()
{
  for (auto& [id, sub_layout] : _sub_layouts) {
    if (sub_layout != nullptr) {
      delete sub_layout;
      sub_layout = nullptr;
    }
  }
  if (_engine != nullptr) {
    delete _engine;
    _engine = nullptr;
  }

  _sub_layouts.clear();
}

bool DrcEngineLayout::addRect(int llx, int lly, int urx, int ury, int net_id)
{
  auto* engine = get_net_engine(net_id);
  if (engine == nullptr) {
    return false;
  }

  engine->addRect(llx, lly, urx, ury);

  return true;
}

DrcEngineSubLayout* DrcEngineLayout::get_sub_layout(int net_id)
{
  auto it = _sub_layouts.find(net_id);
  if (it != _sub_layouts.end()) {
    return it->second;
  } else {
    DrcEngineSubLayout* sub_layout = new DrcEngineSubLayout(net_id);
    _sub_layouts.insert(std::make_pair(net_id, sub_layout));
    return sub_layout;
  }
}
/**
 * clear sub layout marked net id
 */
void DrcEngineLayout::clearSublayoutMark()
{
  for (auto& [net_id, sub_layout] : _sub_layouts) {
    sub_layout->clearChecked();
  }
}

ieda_solver::GeometryBoost* DrcEngineLayout::get_net_engine(int net_id)
{
  auto* sub_layout = get_sub_layout(net_id);

  return sub_layout == nullptr ? nullptr : (ieda_solver::GeometryBoost*) sub_layout->get_engine();
}

// uint64_t DrcEngineLayout::pointCount()
// {
//   uint64_t point_number = 0;
//   for (auto [net_id, sub_layout] : _sub_layouts) {
//     /// build engine data
//     auto* boost_engine = static_cast<ieda_solver::GeometryBoost*>(sub_layout->get_engine());
//     auto boost_pt_list_pair = boost_engine->get_boost_polygons_points();

//     point_number += boost_pt_list_pair.first;  /// boost_pt_list_pair : first value is points number
//   }
//   return point_number;
// }

void DrcEngineLayout::combineLayout()
{
  for (auto& [net_id, sub_layout] : _sub_layouts) {
    _engine->addGeometry(sub_layout->get_engine());

    /// build engine sublayout RTree
    addRTreeSubLayout(sub_layout);
  }
#if 0
  /// save intersect layout for each sublayout
  std::vector<DrcEngineSubLayout*> sub_layouts;
  /// init sublayout drc map & sublayout list
  for (auto& [net_id, sub_layout] : _sub_layouts) {
    sub_layouts.push_back(sub_layout);
  }

  /// sort by bounding box area
  std::sort(sub_layouts.begin(), sub_layouts.end(), [](DrcEngineSubLayout* a, DrcEngineSubLayout* b) {
    auto [llx_a, lly_a, urx_a, ury_a] = a->get_engine()->bounding_box();
    auto [llx_b, lly_b, urx_b, ury_b] = b->get_engine()->bounding_box();
    double area_a = (((double) (urx_a - llx_a)) / 1000) * (((double) (ury_a - lly_a)) / 1000);
    double area_b = (((double) (urx_b - llx_b)) / 1000) * (((double) (ury_b - lly_b)) / 1000);
    return area_a < area_b;
  });

  // #pragma omp parallel for schedule(dynamic)
  for (auto sub_layout : sub_layouts) {
    auto [llx, lly, urx, ury] = sub_layout->get_engine()->bounding_box();
    auto query_sub_layouts = querySubLayouts(llx, lly, urx, ury);

    auto& intersect_layouts = sub_layout->get_intersect_layouts();
    for (auto& [bg_rect, query_sub_layout] : query_sub_layouts) {
      intersect_layouts.insert(std::make_pair(query_sub_layout->get_id(), query_sub_layout));
      if (false == query_sub_layout->hasChecked(sub_layout->get_id()) && false == sub_layout->hasChecked(query_sub_layout->get_id())) {
        query_sub_layout->markChecked(sub_layout->get_id());
      }
    }
  }
#endif
}

void DrcEngineLayout::addRTreeSubLayout(DrcEngineSubLayout* sub_layout)
{
  for (auto rect : sub_layout->get_engine()->getRects()) {
    ieda_solver::BgRect rtree_rect(ieda_solver::BgPoint(boost::polygon::xl(rect), boost::polygon::yl(rect)),
                                   ieda_solver::BgPoint(boost::polygon::xh(rect), boost::polygon::yh(rect)));
    _query_tree.insert(std::make_pair(rtree_rect, sub_layout));
  }
}

std::vector<std::pair<ieda_solver::BgRect, DrcEngineSubLayout*>> DrcEngineLayout::querySubLayouts(int llx, int lly, int urx, int ury)
{
  std::vector<std::pair<ieda_solver::BgRect, DrcEngineSubLayout*>> result;
  ieda_solver::BgRect rect(ieda_solver::BgPoint(llx, lly), ieda_solver::BgPoint(urx, ury));
  _query_tree.query(bg::index::intersects(rect), std::back_inserter(result));

  return result;
}

std::set<int> DrcEngineLayout::querySubLayoutNetId(int llx, int lly, int urx, int ury)
{
  std::set<int> net_ids;

  auto sub_layouts = querySubLayouts(llx, lly, urx, ury);
  for (auto& [bg_rect, sub_layout] : sub_layouts) {
    if (sub_layout == nullptr) {
      continue;
    }
    net_ids.insert(sub_layout->get_id());
    // FIXME: REMOVE this extra check
    // if (!sub_layout->isIntersect(llx, lly, urx, ury)) {
    //   printf("Error: sub_layout is not intersect with query rect\n");
    // }
  }

  return net_ids;
}

}  // namespace idrc