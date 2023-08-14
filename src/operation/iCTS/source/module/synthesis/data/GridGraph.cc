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
#include "GridGraph.h"

#include "log/Log.hh"

namespace icts {
using ieda::Log;

Rectangle GridGraph::findPlacedLocation(Rectangle& start_rect) const
{
  queue<Rectangle> rect_queue;
  set<Rectangle> added_rects;
  rect_queue.push(start_rect);
  added_rects.insert(start_rect);
  while (!rect_queue.empty()) {
    auto cur_rect = rect_queue.front();
    rect_queue.pop();

    if (placeable(cur_rect)) {
      return cur_rect;
    }

    vector<Rectangle> adjacent_rects = adjacentRectangles(cur_rect);
    for (auto& adjacent_rect : adjacent_rects) {
      if (added_rects.count(adjacent_rect) == 0) {
        added_rects.insert(adjacent_rect);
        rect_queue.push(adjacent_rect);
      }
    }
  }
  // can't find the placeable location for rectangle
  LOG_FATAL << "can't find the placeable location for rectangle";
}

bool GridGraph::placeable(const Rectangle& rect) const
{
  if (!withinBoundary(rect)) {
    return false;
  }

  vector<Point> points = getPoints(rect);
  for (auto& point : points) {
    if (!empty(point)) {
      return false;
    }
  }
  return true;
}

vector<Rectangle> GridGraph::adjacentRectangles(const Rectangle& rect, const int& interval) const
{
  vector<Rectangle> rects;
  Rectangle cur_rect;

  Coordinate xl = gtl::xl(rect);
  Coordinate yl = gtl::yl(rect);
  Coordinate xh = gtl::xh(rect);
  Coordinate yh = gtl::yh(rect);
  Coordinate width = xh - xl;
  Coordinate height = yh - yl;

  cur_rect = Rectangle(Point(xl + interval, yl), width, height);
  if (withinBoundary(cur_rect)) {
    rects.emplace_back(cur_rect);
  }
  cur_rect = Rectangle(Point(xl + interval, yl - interval), width, height);
  if (withinBoundary(cur_rect)) {
    rects.emplace_back(cur_rect);
  }
  cur_rect = Rectangle(Point(xl, yl - interval), width, height);
  if (withinBoundary(cur_rect)) {
    rects.emplace_back(cur_rect);
  }
  cur_rect = Rectangle(Point(xl - interval, yl - interval), width, height);
  if (withinBoundary(cur_rect)) {
    rects.emplace_back(cur_rect);
  }
  cur_rect = Rectangle(Point(xl - interval, yl), width, height);
  if (withinBoundary(cur_rect)) {
    rects.emplace_back(cur_rect);
  }
  cur_rect = Rectangle(Point(xl - interval, yl + interval), width, height);
  if (withinBoundary(cur_rect)) {
    rects.emplace_back(cur_rect);
  }
  cur_rect = Rectangle(Point(xl, yl + interval), width, height);
  if (withinBoundary(cur_rect)) {
    rects.emplace_back(cur_rect);
  }
  cur_rect = Rectangle(Point(xl + interval, yl + interval), width, height);
  if (withinBoundary(cur_rect)) {
    rects.emplace_back(cur_rect);
  }
  if (rects.empty()) {
    return adjacentRectangles(rect, interval + 1);
  } else {
    return rects;
  }
}

void GridGraph::setBlockage(const Rectangle& rect)
{
  vector<Point> points = getPoints(rect);
  for (auto& point : points) {
    setBlockage(point);
  }
}

void GridGraph::setBlockage(const Point& point)
{
  if (withinBoundary(point)) {
    auto x = point.x();
    auto y = point.y();
    _grids[y][x] = 1;
  }
}

void GridGraph::resetBlockage(const Rectangle& rect)
{
  std::ranges::for_each(getPoints(rect), [&](const Point& point) { resetBlockage(point); });
}

void GridGraph::resetBlockage(const Point& point)
{
  auto x = point.x();
  auto y = point.y();
  _grids[y][x] = 0;
}

vector<Point> GridGraph::getPoints(const Rectangle& rect) const
{
  vector<Point> points;

  auto horizon_interval = rect.get(gtl::HORIZONTAL);
  auto vertical_interval = rect.get(gtl::VERTICAL);

  for (Coordinate y = vertical_interval.low(); y <= vertical_interval.high(); ++y) {
    for (Coordinate x = horizon_interval.low(); x <= horizon_interval.high(); ++x) {
      points.emplace_back(Point(x, y));
    }
  }
  return points;
}

Point GridGraph::legalization(const Point& point) const
{
  Point legal_loc = point;
  if (!withinBoundary(point)) {
    if (point.x() < 0) {
      legal_loc.x(0);
    } else {
      legal_loc.x(_col_num);
    }
    if (point.y() < 0) {
      legal_loc.y(0);
    } else {
      legal_loc.y(_row_num);
    }
  }
  return legal_loc;
}

}  // namespace icts