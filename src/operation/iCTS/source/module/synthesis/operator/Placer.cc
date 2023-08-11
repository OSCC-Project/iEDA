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
#include "Placer.h"

#include "CTSAPI.hpp"

namespace icts {

void Placer::init()
{
  CTSAPIInst.saveToLog("\n\nPlacer Log");
  auto db_wrapper = CTSAPIInst.get_db_wrapper();

  _grid_graph = GridGraph(db_wrapper->get_row_num(), db_wrapper->get_site_num());

  Rectangle core_box = db_wrapper->get_core_bounding_box();
  _spacing_x = gtl::xl(core_box);
  _spacing_y = gtl::yl(core_box);

  _grid_width = db_wrapper->get_site_width();
  _grid_height = db_wrapper->get_site_height();
  CTSAPIInst.saveToLog("Die region: [", core_box.low().x(), ", ", core_box.low().y(), "] - [", core_box.high().x(), ", ",
                       core_box.high().y(), "]\n\n");
  placeBlockages();
}

void Placer::placeBlockages()
{
  auto db_wrapper = CTSAPIInst.get_db_wrapper();
  auto rects = db_wrapper->get_blockages();
  for (auto& rect : rects) {
    setBlockage(rect);
  }
}

void Placer::placeInstance(CtsInstance* inst)
{
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  // get the bounding box of instance
  Rectangle inst_bounding_box = db_wrapper->get_bounding_box(inst);

  // find suitable location in grid coordination
  auto new_bounding_box = findPlacedLocation(inst_bounding_box);

  // place instance to suitable location
  auto loc = Point(gtl::xl(new_bounding_box), gtl::yl(new_bounding_box));

  inst->set_location(loc);

  // set blocakage to grid graph
  setBlockage(new_bounding_box);
}

void Placer::cancelPlaceInstance(CtsInstance* inst)
{
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  // get the bounding box of instance
  Rectangle inst_bounding_box = db_wrapper->get_bounding_box(inst);
  resetBlockage(inst_bounding_box);
}

Rectangle Placer::findPlacedLocation(const Rectangle& rect) const
{
  auto grid_rect = coordToGrid(rect);
  auto lx = grid_rect.low().x();
  auto ly = grid_rect.low().y();
  if (lx < 0) {
    gtl::move(grid_rect, gtl::HORIZONTAL, std::abs(lx));
  }
  if (ly < 0) {
    gtl::move(grid_rect, gtl::VERTICAL, std::abs(ly));
  }
  auto placed_grid_rect = _grid_graph.findPlacedLocation(grid_rect);
  return gridToCoord(placed_grid_rect);
}

Rectangle Placer::coordToGrid(const Rectangle& rect) const
{
  auto min_coord = coordToGrid(Point(gtl::xl(rect), gtl::yl(rect)));
  auto max_coord = coordToGrid(Point(gtl::xh(rect), gtl::yh(rect)));
  if (min_coord == max_coord) {
    return Rectangle(min_coord.x(), min_coord.y(), max_coord.x(), max_coord.y());
  } else {
    return Rectangle(min_coord.x(), min_coord.y(), max_coord.x() - 1, max_coord.y() - 1);
  }
}

Point Placer::coordToGrid(const Point& point) const
{
  int x = (point.x() - _spacing_x) / _grid_width;
  int y = (point.y() - _spacing_y) / _grid_height;
  return Point(x, y);
}

Rectangle Placer::gridToCoord(const Rectangle& rect) const
{
  auto min_coord = gridToCoord(Point(gtl::xl(rect), gtl::yl(rect)));
  auto max_coord = gridToCoord(Point(gtl::xh(rect), gtl::yh(rect)));
  return Rectangle(min_coord.x(), min_coord.y(), max_coord.x(), max_coord.y());
}

Point Placer::gridToCoord(const Point& point) const
{
  int x = point.x() * _grid_width + _spacing_x;
  int y = point.y() * _grid_height + _spacing_y;
  return Point(x, y);
}

void Placer::setBlockage(const Rectangle& rect)
{
  Rectangle grid_rect = coordToGrid(rect);
  _grid_graph.setBlockage(grid_rect);
}

void Placer::resetBlockage(const Rectangle& rect)
{
  Rectangle grid_rect = coordToGrid(rect);
  _grid_graph.resetBlockage(grid_rect);
}

bool Placer::isInCore(const Point& location) const
{
  auto grid_loc = coordToGrid(location);
  return _grid_graph.withinBoundary(grid_loc);
}

Point Placer::legalization(const Point& location) const
{
  auto grid_loc = coordToGrid(location);
  auto legal_grid_loc = _grid_graph.legalization(grid_loc);
  return gridToCoord(legal_grid_loc);
}

}  // namespace icts