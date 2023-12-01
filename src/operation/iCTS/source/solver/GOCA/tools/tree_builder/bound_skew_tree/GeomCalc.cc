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
/**
 * @file GeomCalc.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "GeomCalc.hh"

#include "log/Log.hh"
namespace icts {
namespace bst {
bool GeomCalc::isSame(const Pt& p1, const Pt& p2)
{
  auto dist = distance(p1, p2);
  return Equal(dist, 0);
}

double GeomCalc::distance(const Pt& p1, const Pt& p2)
{
  return std::abs(p1.x - p2.x) + std::abs(p1.y - p2.y);
}

double GeomCalc::ptToLineDistManhattan(Pt& p, const Line& l, Pt& closest)
{
  // manhattan arc
  Trr ms_p = Trr(p, 0);
  Trr ms_l;
  lineToMs(ms_l, l);
  auto dist = msDistance(ms_p, ms_l);
  Trr new_ms_p;
  new_ms_p.makeDiamond(p, dist);
  Trr intersect_ms;
  makeIntersect(new_ms_p, ms_l, intersect_ms);
  coreMidPoint(intersect_ms, closest);
  return dist;
}

double GeomCalc::ptToLineDistNotManhattan(Pt& p, const Line& l, Pt& closest)
{
  std::vector<Pt> candidate;
  if (!Equal(l[kHead].x, l[kTail].x) && (p.x - l[kHead].x) * (p.x - l[kTail].x) <= 0) {
    Pt pt;
    pt.x = p.x;
    pt.y = (l[kTail].x - p.x) * (l[kHead].y - l[kTail].y) / (l[kTail].x - l[kHead].x) + l[kTail].y;
    candidate.push_back(pt);
  }
  if (!Equal(l[kHead].y, l[kTail].y) && (p.y - l[kHead].y) * (p.y - l[kTail].y) <= 0) {
    Pt pt;
    pt.x = (l[kTail].y - p.y) * (l[kHead].x - l[kTail].x) / (l[kTail].y - l[kHead].y) + l[kTail].x;
    pt.y = p.y;
    candidate.push_back(pt);
  }
  if (candidate.size() < 2) {
    candidate.push_back(l[kHead]);
    candidate.push_back(l[kTail]);
  }
  double min_dist = std::numeric_limits<double>::max();
  std::ranges::for_each(candidate, [&p, &min_dist, &closest](auto& pt) {
    auto dist = distance(p, pt);
    if (dist < min_dist) {
      min_dist = dist;
      closest = pt;
    }
  });
  return min_dist;
}

double GeomCalc::ptToLineDist(Pt& p, const Line& l, Pt& closest)
{
  auto min_dist = std::numeric_limits<double>::max();
  auto delta_x = std::abs(l[kHead].x - l[kTail].x);
  auto delta_y = std::abs(l[kHead].y - l[kTail].y);
  if (isSame(l[kHead], l[kTail])) {
    closest = l[kHead];
    min_dist = distance(p, closest);
  } else if (onLine(p, l)) {
    closest = p;
    min_dist = 0;
  } else if (Equal(delta_x, delta_y)) {
    // manhattan arc
    min_dist = ptToLineDistManhattan(p, l, closest);
  } else {
    // not manhattan arc
    min_dist = ptToLineDistNotManhattan(p, l, closest);
  }
  LOG_FATAL_IF(onLine(closest, l) == false) << "closest point is not on line";
  return min_dist;
}

double GeomCalc::ptToTrrDist(Pt& p, Trr& ms)
{
  Trr pt_trr(p, 0);
  if (isTrrContain(pt_trr, ms)) {
    return 0;
  }
  std::vector<Trr> trrs(4, ms);
  trrs[kLeft + kHead].y_low(ms.y_high());
  trrs[kLeft + kTail].y_high(ms.y_low());
  trrs[kRight + kHead].x_low(ms.x_high());
  trrs[kRight + kTail].x_high(ms.x_low());
  double min_dist = std::numeric_limits<double>::max();
  std::ranges::for_each(trrs, [&pt_trr, &min_dist](auto& trr) {
    auto dist = msDistance(pt_trr, trr);
    min_dist = std::min(min_dist, dist);
  });
  return min_dist;
}

void GeomCalc::calcCoord(Pt& p, const Line& l, const double& shift)
{
  auto line_type = lineType(l);
  double d0 = 0;
  double d1 = 0;
  if (line_type == LineType::kHorizontal || line_type == LineType::kFlat) {
    d0 = std::abs(l[kHead].x - shift);
    d1 = std::abs(l[kTail].x - shift);
  } else {
    d0 = std::abs(l[kHead].y - shift);
    d1 = std::abs(l[kTail].y - shift);
  }
  p.x = (l[kHead].x * d1 + l[kTail].x * d0) / (d0 + d1);
  p.y = (l[kHead].y * d1 + l[kTail].y * d0) / (d0 + d1);
}

void GeomCalc::calcRelativeCoord(Pt& p, const RelativeType& type, const double& shift)
{
  switch (type) {
    case RelativeType::kLeft:
      p.x += shift;
      break;
    case RelativeType::kRight:
      p.x -= shift;
      break;
    case RelativeType::kTop:
      p.y -= shift;
      break;
    case RelativeType::kBottom:
      p.y += shift;
      break;
    default:
      break;
  }
}

double GeomCalc::crossProduct(const Pt& p1, const Pt& p2, const Pt& p3)
{
  return (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y);
};

LineType GeomCalc::lineType(const Line& l)
{
  return lineType(l[kHead], l[kTail]);
}

LineType GeomCalc::lineType(const Pt& p1, const Pt& p2)
{
  auto d_x = std::abs(p1.x - p2.x);
  auto d_y = std::abs(p1.y - p2.y);
  if (Equal(d_x, d_y)) {
    return LineType::kManhattan;
  } else if (Equal(d_x, 0)) {
    return LineType::kVertical;
  } else if (Equal(d_y, 0)) {
    return LineType::kHorizontal;
  } else if (d_x > d_y) {
    return LineType::kFlat;
  } else {
    return LineType::kTilt;
  }
  LOG_FATAL << "line type error" << std::endl;
}

IntersectType GeomCalc::lineIntersect(Pt& p, Line& l1, Line& l2)
{
  LOG_FATAL_IF(Equal(distance(l1[kHead], l1[kTail]), 0) || Equal(distance(l2[kHead], l2[kTail]), 0)) << "line length is zero";
  if (!boundBoxOverlap(l1, l2)) {
    return IntersectType::kNone;
  }
  if (distance(l1[kHead], l2[kHead]) + distance(l1[kTail], l2[kTail]) < kEpsilon) {
    p = l1[kHead];
    return IntersectType::kSame;
  }
  size_t count = 0;
  FOR_EACH_SIDE(side)
  {
    if (onLine(l1[side], l2)) {
      p = l1[side];
      ++count;
    }
  }
  FOR_EACH_SIDE(side)
  {
    if (onLine(l2[side], l1)) {
      p = l2[side];
      ++count;
    }
  }
  FOR_EACH_SIDE(left_side)
  {
    FOR_EACH_SIDE(right_side)
    {
      if (distance(l1[left_side], l2[right_side]) < kEpsilon) {
        --count;
      }
    }
  }

  if (count >= 2) {
    return IntersectType::kOverlap;
  }

  auto l1_x_is_equal = Equal(l1[kHead].x, l1[kTail].x);
  auto l2_x_is_equal = Equal(l2[kHead].x, l2[kTail].x);
  if (l1_x_is_equal && l2_x_is_equal) {
    // parallel vertical lines
  } else if (l1_x_is_equal && !l2_x_is_equal) {
    p.x = l1[kHead].x;
    p.y = (l2[kHead].y - l2[kTail].y) * (l1[kHead].x - l2[kHead].x) / (l2[kHead].x - l2[kTail].x) + l2[kHead].y;
  } else if (!l1_x_is_equal && l2_x_is_equal) {
    p.x = l2[kHead].x;
    p.y = (l1[kTail].y - l1[kHead].y) * (l2[kHead].x - l1[kHead].x) / (l1[kTail].x - l1[kHead].x) + l1[kHead].y;
  } else if (!l1_x_is_equal && !l2_x_is_equal) {
    auto l1_k = (l1[kTail].y - l1[kHead].y) / (l1[kTail].x - l1[kHead].x);
    auto l2_k = (l2[kTail].y - l2[kHead].y) / (l2[kTail].x - l2[kHead].x);
    if (Equal(l1_k, l2_k)) {
      // parallel lines
      return IntersectType::kNone;
    } else {
      p.x = (l2[kHead].y - l1[kHead].y + l1[kHead].x * l1_k - l2[kHead].x * l2_k) / (l1_k - l2_k);
      p.y = (l1[kHead].y + l2[kHead].y + l1_k * (p.x - l1[kHead].x) + l2_k * (p.x - l2[kHead].x)) / 2.0;
    }
  }
  if (inBoundBox(p, l1) && inBoundBox(p, l2)) {
    return IntersectType::kCrossing;
  }
  return IntersectType::kNone;
}

RelativeType GeomCalc::lineRelative(const Line& l1, const Line& l2, const size_t& ref)
{
  // return the position of one line relative to ref line
  auto line_type = lineType(l1);
  auto t_line_type = lineType(l2);
  LOG_FATAL_IF(line_type != t_line_type) << "line type is not same";
  if (line_type == LineType::kVertical || line_type == LineType::kTilt) {
    auto max_x1 = std::max(l1[kHead].x, l1[kTail].x);
    auto max_x2 = std::max(l2[kHead].x, l2[kTail].x);
    if ((max_x1 <= max_x2 && ref == kLeft) || (max_x1 >= max_x2 && ref == kRight)) {
      return RelativeType::kLeft;
    } else {
      return RelativeType::kRight;
    }
  } else if (line_type == LineType::kHorizontal || line_type == LineType::kFlat) {
    auto max_y1 = std::max(l1[kHead].y, l1[kTail].y);
    auto max_y2 = std::max(l2[kHead].y, l2[kTail].y);
    if ((max_y1 <= max_y2 && ref == kLeft) || (max_y1 >= max_y2 && ref == kRight)) {
      return RelativeType::kBottom;
    } else {
      return RelativeType::kTop;
    }
  } else if (line_type == LineType::kManhattan) {
    return RelativeType::kManhattanParallel;
  } else {
    LOG_FATAL << "line type error" << std::endl;
  }
}

double GeomCalc::lineDist(Line& l1, Line& l2, PtPair& closest)
{
  double dist, min_dist = std::numeric_limits<double>::max();
  Pt intersect;
  Side<Side<Pt>> pt;
  pt[kLeft] = l1;
  pt[kRight] = l2;
  size_t n1 = 2;
  size_t n2 = 2;
  if (isSame(l1[kHead], l1[kTail])) {
    n1 = 1;
  }
  if (isSame(l2[kHead], l2[kTail])) {
    n2 = 1;
  }
  if (n1 == 2 && n2 == 2) {
    if (lineIntersect(intersect, l1, l2) != IntersectType::kNone) {
      closest[kLeft] = closest[kRight] = intersect;
      return 0;
    }
  }
  for (size_t i = 0; i < n1; ++i) {
    size_t k = (i + 1) % 2;
    for (size_t j = 0; j < n2; ++j) {
      dist = ptToLineDist(pt[i][j], pt[k], intersect);
      if (dist < min_dist) {
        min_dist = dist;
        closest[i] = pt[i][j];
        closest[k] = intersect;
      }
    }
  }
  return min_dist;
}

bool GeomCalc::onLine(Pt& p, const Line& l)
{
  auto len = distance(l[kHead], l[kTail]);
  auto len_to_head = distance(p, l[kHead]);
  auto len_to_tail = distance(p, l[kTail]);
  if (std::abs(len_to_head + len_to_tail - len) < 2 * kEpsilon) {
    if (Equal(len_to_head, 0)) {
      p = l[kHead];
      return true;
    } else if (Equal(len_to_tail, 0)) {
      p = l[kTail];
      return true;
    } else {
      auto delta_x = std::abs(l[kTail].x - l[kHead].x);
      auto delta_y = std::abs(l[kTail].y - l[kHead].y);
      if (delta_y > delta_x) {
        auto temp_x = (l[kTail].x - l[kHead].x) * (p.y - l[kHead].y) / (l[kTail].y - l[kHead].y) + l[kHead].x;
        if (Equal(temp_x, p.x)) {
          p.x = temp_x;
          return true;
        }
      } else {
        auto temp_y = (l[kTail].y - l[kHead].y) * (p.x - l[kHead].x) / (l[kTail].x - l[kHead].x) + l[kHead].y;
        if (Equal(temp_y, p.y)) {
          p.y = temp_y;
          return true;
        }
      }
    }
  }
  return false;
}

bool GeomCalc::isParallel(const Line& l1, const Line& l2)
{
  if (isSame(l1[kHead], l1[kTail]) || isSame(l2[kHead], l2[kTail])) {
    return false;
  }
  auto delta_x1 = std::abs(l1[kHead].x - l1[kTail].x);
  auto delta_y1 = std::abs(l1[kHead].y - l1[kTail].y);
  auto delta_x2 = std::abs(l2[kHead].x - l2[kTail].x);
  auto delta_y2 = std::abs(l2[kHead].y - l2[kTail].y);
  if (Equal(delta_x1, 0) && Equal(delta_x2, 0)) {
    return true;
  } else if (!Equal(delta_x1, 0) && !Equal(delta_x2, 0) && Equal(delta_y1 / delta_x1, delta_y2 / delta_x2)) {
    return true;
  }
  return false;
}

bool GeomCalc::inBoundBox(const Pt& p, const Line& l)
{
  if (p.x <= std::max(l[kHead].x, l[kTail].x) + kEpsilon && p.x >= std::min(l[kHead].x, l[kTail].x) - kEpsilon
      && p.y <= std::max(l[kHead].y, l[kTail].y) + kEpsilon && p.y >= std::min(l[kHead].y, l[kTail].y) - kEpsilon) {
    return true;
  }
  return false;
}

bool GeomCalc::boundBoxOverlap(const Line& l1, const Line& l2, const double& epsilon)
{
  return boundBoxOverlap(l1[kHead].x, l1[kHead].y, l1[kTail].x, l1[kTail].y, l2[kHead].x, l2[kHead].y, l2[kTail].x, l2[kTail].y, epsilon);
}

bool GeomCalc::boundBoxOverlap(const double& x1, const double& y1, const double& x2, const double& y2, const double& x3, const double& y3,
                               const double& x4, const double& y4, const double& epsilon)
{
  if (std::min(x1, x2) - std::max(x3, x4) >= epsilon) {
    return false;
  }
  if (std::min(x3, x4) - std::max(x1, x2) >= epsilon) {
    return false;
  }
  if (std::min(y1, y2) - std::max(y3, y4) >= epsilon) {
    return false;
  }
  if (std::min(y3, y4) - std::max(y1, y2) >= epsilon) {
    return false;
  }
  return true;
}

double GeomCalc::msDistance(Trr& ms1, Trr& ms2)
{
  checkMs(ms1);
  checkMs(ms2);
  // Trr<T> is in a Linfinity metric space
  auto x1_low = ms1.x_low();
  auto x1_high = ms1.x_high();
  auto y1_low = ms1.y_low();
  auto y1_high = ms1.y_high();
  auto x2_low = ms2.x_low();
  auto x2_high = ms2.x_high();
  auto y2_low = ms2.y_low();
  auto y2_high = ms2.y_high();
  /*
  (1) Not intersect between x-coords and y-coords
  (2) Not intersect between x-coords, but intersect between y-coords
  (3) Not intersect between y-coords, but intersect between x-coords
  (4) Intersect between x-coords and y-coords (distance = 0)
  */
  auto x_is_intersect = (x1_low <= x2_high) && (x2_low <= x1_high);
  auto y_is_intersect = (y1_low <= y2_high) && (y2_low <= y1_high);
  if (!x_is_intersect && !y_is_intersect) {
    // (1)
    auto low1_to_low2 = std::max(std::abs(x1_low - x2_low), std::abs(y1_low - y2_low));
    auto low1_to_high2 = std::max(std::abs(x1_low - x2_high), std::abs(y1_low - y2_high));
    auto high1_to_low2 = std::max(std::abs(x1_high - x2_low), std::abs(y1_high - y2_low));
    auto high1_to_high2 = std::max(std::abs(x1_high - x2_high), std::abs(y1_high - y2_high));
    return std::min(std::min(low1_to_low2, low1_to_high2), std::min(high1_to_low2, high1_to_high2));
  } else if (!x_is_intersect) {
    // (2)
    auto low1_to_high2 = x1_low - x2_high;
    auto low2_to_high1 = x2_low - x1_high;
    return std::max(low1_to_high2, low2_to_high1);
  } else if (!y_is_intersect) {
    // (3)
    auto low1_to_high2 = y1_low - y2_high;
    auto low2_to_high1 = y2_low - y1_high;
    return std::max(low1_to_high2, low2_to_high1);
  } else {
    // (4)
    return 0;
  }
}

void GeomCalc::makeIntersect(Trr& ms1, Trr& ms2, Trr& intersect)
{
  intersect.x_low(std::max(ms1.x_low(), ms2.x_low()));
  intersect.x_high(std::min(ms1.x_high(), ms2.x_high()));
  intersect.y_low(std::max(ms1.y_low(), ms2.y_low()));
  intersect.y_high(std::min(ms1.y_high(), ms2.y_high()));
  checkMs(intersect);
}

void GeomCalc::coreMidPoint(Trr& ms, Pt& mid)
{
  auto x = (ms.x_low() + ms.x_high()) / 2;
  auto y = (ms.y_low() + ms.y_high()) / 2;
  mid.x = (x + y) / 2;
  mid.y = (y - x) / 2;
}

bool GeomCalc::isTrrContain(const Trr& small, const Trr& huge)
{
  if (small.x_high() <= huge.x_high() + kEpsilon && small.x_low() >= huge.x_low() - kEpsilon && small.y_high() <= huge.y_high() + kEpsilon
      && small.y_low() >= huge.y_low() - kEpsilon) {
    return true;
  }
  return false;
}

void GeomCalc::buildTrr(const Trr& ms, const double& r, Trr& build_trr)
{
  build_trr.x_low(ms.x_low() - r);
  build_trr.x_high(ms.x_high() + r);
  build_trr.y_low(ms.y_low() - r);
  build_trr.y_high(ms.y_high() + r);
}

void GeomCalc::trrCore(const Trr& trr, Trr& core)
{
  if (trr.x_high() - trr.x_low() < trr.y_high() - trr.y_low()) {
    core.y_low(trr.y_low());
    core.y_high(trr.y_high());
    auto x = (trr.x_low() + trr.x_high()) / 2;
    core.x_low(x);
    core.x_high(x);
  } else {
    core.x_low(trr.x_low());
    core.x_high(trr.x_high());
    auto y = (trr.y_low() + trr.y_high()) / 2;
    core.y_low(y);
    core.y_high(y);
  }
}

void GeomCalc::trrToPt(const Trr& trr, Pt& pt)
{
  pt.x = (trr.y_low() + trr.x_high()) / 2;
  pt.y = (trr.y_low() - trr.x_high()) / 2;
}

void GeomCalc::trrToRegion(Trr& trr, Region& region)
{
  auto x = trr.x_high() - trr.x_low();
  auto y = trr.y_high() - trr.y_low();
  if (Equal(x, 0) && Equal(y, 0)) {
    Pt pt;
    trrToPt(trr, pt);
    region.push_back(pt);
    return;
  } else if (Equal(x, 0) || Equal(y, 0)) {
    Pt head, tail;
    msToLine(trr, head, tail);
    region.push_back(head);
    region.push_back(tail);
    return;
  }

  Trr ms(trr.x_high(), trr.x_high(), trr.y_low(), trr.y_high());
  Pt head, tail;

  msToLine(ms, head, tail);
  region.push_back(head);
  region.push_back(tail);

  ms.x_low(trr.x_low());
  ms.x_high(trr.x_low());

  msToLine(ms, head, tail);
  region.push_back(head);
  region.push_back(tail);
}

bool GeomCalc::isSegmentTrr(const Trr& trr)
{
  return Equal(trr.x_low(), trr.x_high()) || Equal(trr.y_low(), trr.y_high());
}

void GeomCalc::sortPtsByFront(Pts& pts)
{
  // sort by dist to first point
  std::ranges::for_each(pts, [&pts](Pt& p) { p.val = distance(p, pts.front()); });
  sortPtsByVal(pts);
}

void GeomCalc::sortPtsByVal(Pts& pts)
{
  if (pts.empty()) {
    return;
  }
  std::ranges::sort(pts, [](const Pt& p1, const Pt& p2) { return p1.val < p2.val; });
}

void GeomCalc::sortPtsByValDec(Pts& pts)
{
  if (pts.empty()) {
    return;
  }
  std::ranges::sort(pts, [](const Pt& p1, const Pt& p2) { return p1.val > p2.val; });
}

void GeomCalc::uniquePtsLoc(std::vector<Pt>& pts)
{
  if (pts.size() < 2) {
    return;
  }
  std::vector<Pt> unique_pts = {pts.front()};
  std::ranges::for_each(pts, [&unique_pts](const Pt& p) {
    if (!isSame(p, unique_pts.back())) {
      unique_pts.push_back(p);
    }
  });
  if (unique_pts.size() > 1 && isSame(unique_pts.front(), unique_pts.back())) {
    unique_pts.pop_back();
  }
  pts = unique_pts;
}

void GeomCalc::uniquePtsVal(std::vector<Pt>& pts)
{
  pts.erase(std::unique(pts.begin(), pts.end(), [](const Pt& p1, const Pt& p2) { return Equal(p1.val, p2.val); }), pts.end());
}

std::vector<Line> GeomCalc::getLines(const std::vector<Pt>& pts)
{
  if (pts.size() == 2) {
    return {{pts.front(), pts.back()}};
  }
  std::vector<Line> lines;
  for (size_t i = 0; i < pts.size(); ++i) {
    auto j = (i + 1) % pts.size();
    lines.push_back({pts[i], pts[j]});
  }
  return lines;
}

void GeomCalc::convexHull(std::vector<Pt>& pts)
{
  // check pts num
  if (pts.size() < 2) {
    return;
  }
  if (pts.size() == 2) {
    auto dist = distance(pts.front(), pts.back());
    if (Equal(dist, 0)) {
      pts.pop_back();
    }
    return;
  }
  // calculate convex hull by Andrew algorithm
  std::ranges::sort(pts, [](const Pt& p1, const Pt& p2) { return p1.x + kEpsilon < p2.x || (Equal(p1.x, p2.x) && p1.y < p2.y); });
  std::vector<Pt> ans(2 * pts.size());
  size_t k = 0;
  for (size_t i = 0; i < pts.size(); ++i) {
    while (k > 1 && crossProduct(ans[k - 2], ans[k - 1], pts[i]) <= kEpsilon) {
      --k;
    }
    ans[k++] = pts[i];
  }
  for (size_t i = pts.size() - 1, t = k + 1; i > 0; --i) {
    while (k >= t && crossProduct(ans[k - 2], ans[k - 1], pts[i - 1]) <= kEpsilon) {
      --k;
    }
    ans[k++] = pts[i - 1];
  }
  pts = {ans.begin(), ans.begin() + k - 1};
}

Pt GeomCalc::centerPt(const std::vector<Pt>& pts)
{
  if (pts.empty()) {
    return {0, 0};
  }
  double x = 0, y = 0;
  std::ranges::for_each(pts, [&x, &y](const Pt& p) {
    x += p.x;
    y += p.y;
  });
  return {x / pts.size(), y / pts.size()};
}

bool GeomCalc::isRegionContain(const Pt& p, const std::vector<Pt>& region)
{
  auto is_in_region = false;
  auto p_x = p.x;
  auto p_y = p.y;
  auto n = region.size();
  auto j = n - 1;
  for (size_t i = 0; i < n; j = i, ++i) {
    auto s_x = region[i].x;
    auto s_y = region[i].y;
    auto t_x = region[j].x;
    auto t_y = region[j].y;
    if ((s_y < p_y && t_y >= p_y) || (t_y < p_y && s_y >= p_y)) {
      if (s_x + (p_y - s_y) / (t_y - s_y) * (t_x - s_x) < p_x) {
        is_in_region = !is_in_region;
      }
    }
  }
  if (is_in_region) {
    return true;
  }
  auto pt = p;
  for (size_t i = 0; i < region.size(); ++i) {
    auto j = (i + 1) % region.size();
    if (onLine(pt, {region[i], region[j]})) {
      return true;
    }
  }
  return false;
}

Pt GeomCalc::closestPtOnRegion(const Pt& p, const std::vector<Pt>& region)
{
  if (isRegionContain(p, region)) {
    return p;
  }
  auto pt = p;
  Pt closest;
  Pt ans;
  auto min_dist = std::numeric_limits<double>::max();
  for (size_t i = 0; i < region.size(); ++i) {
    auto j = (i + 1) % region.size();
    auto dist = ptToLineDist(pt, {region[i], region[j]}, closest);
    if (dist < min_dist) {
      min_dist = dist;
      ans = closest;
    }
  }
  return ans;
}

void GeomCalc::lineToMs(Trr& ms, const Line& l)
{
  lineToMs(ms, l[kHead], l[kTail]);
}

void GeomCalc::lineToMs(Trr& ms, const Pt& p1, const Pt& p2)
{
  if (p1.y <= p2.y) {
    ms.x_low(p2.x - p2.y);
    ms.x_high(p1.x - p1.y);
    ms.y_low(p1.x + p1.y);
    ms.y_high(p2.x + p2.y);
  } else {
    ms.x_low(p1.x - p1.y);
    ms.x_high(p2.x - p2.y);
    ms.y_low(p2.x + p2.y);
    ms.y_high(p1.x + p1.y);
  }
  checkMs(ms);
}

void GeomCalc::msToLine(Trr& ms, Line& l)
{
  msToLine(ms, l[kHead], l[kTail]);
}

void GeomCalc::msToLine(Trr& ms, Pt& p1, Pt& p2)
{
  checkMs(ms);
  p1.x = (ms.y_low() + ms.x_high()) / 2;
  p1.y = (ms.y_low() - ms.x_high()) / 2;
  p2.x = (ms.y_high() + ms.x_low()) / 2;
  p2.y = (ms.y_high() - ms.x_low()) / 2;
  LOG_FATAL_IF(p1.y > p2.y) << "p2.y should be larger than p1.y";
}

void GeomCalc::checkMs(Trr& ms)
{
  auto x_low = ms.x_low();
  auto x_high = ms.x_high();
  if (Equal(x_low, x_high)) {
    auto avg = (x_low + x_high) / 2;
    ms.x_low(avg);
    ms.x_high(avg);
  }
  auto y_low = ms.y_low();
  auto y_high = ms.y_high();
  if (Equal(y_low, y_high)) {
    auto avg = (y_low + y_high) / 2;
    ms.y_low(avg);
    ms.y_high(avg);
  }
  LOG_FATAL_IF(ms.x_low() > ms.x_high() || ms.y_low() > ms.y_high()) << "ms is not valid";
}

}  // namespace bst
}  // namespace icts