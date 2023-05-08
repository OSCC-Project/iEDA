#include "Placer.h"

#include "CTSAPI.hpp"
namespace icts {

void Placer::init() {
  auto db_wrapper = CTSAPIInst.get_db_wrapper();

  _grid_graph =
      GridGraph(db_wrapper->get_row_num(), db_wrapper->get_site_num());

  Rectangle core_box = db_wrapper->get_core_bounding_box();
  _spacing_x = gtl::xl(core_box);
  _spacing_y = gtl::yl(core_box);

  _grid_width = db_wrapper->get_site_width();
  _grid_height = db_wrapper->get_site_height();

  placeBlockages();
}

void Placer::placeBlockages() {
  auto db_wrapper = CTSAPIInst.get_db_wrapper();
  auto rects = db_wrapper->get_blockages();
  for (auto &rect : rects) {
    setBlockage(rect);
  }
}

void Placer::placeInstance(CtsInstance *inst) {
  auto *db_wrapper = CTSAPIInst.get_db_wrapper();
  // get the bounding box of instance
  Rectangle inst_bounding_box = db_wrapper->get_bounding_box(inst);

  // find suitable location in grid coordination
  auto new_bounding_box = findPlacedLocation(inst_bounding_box);

  // place instance to suitable location
  inst->set_location(
      Point(gtl::xl(new_bounding_box), gtl::yl(new_bounding_box)));

  // set blocakage to grid graph
  setBlockage(new_bounding_box);
}

Rectangle Placer::findPlacedLocation(const Rectangle &rect) const {
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

Rectangle Placer::coordToGrid(const Rectangle &rect) const {
  auto min_coord = coordToGrid(Point(gtl::xl(rect), gtl::yl(rect)));
  auto max_coord = coordToGrid(Point(gtl::xh(rect), gtl::yh(rect)));
  return Rectangle(min_coord.x(), min_coord.y(), max_coord.x(), max_coord.y());
}

Point Placer::coordToGrid(const Point &point) const {
  int x = (point.x() - _spacing_x) / _grid_width;
  int y = (point.y() - _spacing_y) / _grid_height;
  return Point(x, y);
}

Rectangle Placer::gridToCoord(const Rectangle &rect) const {
  auto min_coord = gridToCoord(Point(gtl::xl(rect), gtl::yl(rect)));
  auto max_coord = gridToCoord(Point(gtl::xh(rect), gtl::yh(rect)));
  return Rectangle(min_coord.x(), min_coord.y(), max_coord.x(), max_coord.y());
}

Point Placer::gridToCoord(const Point &point) const {
  int x = point.x() * _grid_width + _spacing_x;
  int y = point.y() * _grid_height + _spacing_y;
  return Point(x, y);
}

void Placer::setBlockage(const Rectangle &rect) {
  Rectangle grid_rect = coordToGrid(rect);
  _grid_graph.setBlockage(grid_rect);
}

bool Placer::isInCore(const Point &location) const {
  auto grid_loc = coordToGrid(location);
  return _grid_graph.withinBoundary(grid_loc);
}

Point Placer::legalization(const Point &location) const {
  auto grid_loc = coordToGrid(location);
  auto legal_grid_loc = _grid_graph.legalization(grid_loc);
  return gridToCoord(legal_grid_loc);
}

}  // namespace icts