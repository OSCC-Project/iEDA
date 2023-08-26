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

double GeomCalc::ptToLineDistManhattan(Pt& p, const Line& l, Pt& intersect)
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
  coreMidPoint(intersect_ms, intersect);
  return dist;
}

double GeomCalc::ptToLineDistNotManhattan(Pt& p, const Line& l, Pt& intersect)
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
  std::ranges::for_each(candidate, [&p, &min_dist, &intersect](auto& pt) {
    auto dist = distance(p, pt);
    if (dist < min_dist) {
      min_dist = dist;
      intersect = pt;
    }
  });
  return min_dist;
}

double GeomCalc::ptToLineDist(Pt& p, const Line& l, Pt& intersect)
{
  auto min_dist = std::numeric_limits<double>::max();
  auto delta_x = std::abs(l[kHead].x - l[kTail].x);
  auto delta_y = std::abs(l[kHead].y - l[kTail].y);
  if (isSame(l[kHead], l[kTail])) {
    intersect = l[kHead];
    min_dist = distance(p, intersect);
  } else if (onLine(p, l)) {
    intersect = p;
    min_dist = 0;
  } else if (Equal(delta_x, delta_y)) {
    // manhattan arc
    min_dist = ptToLineDistManhattan(p, l, intersect);
  } else {
    // not manhattan arc
    min_dist = ptToLineDistNotManhattan(p, l, intersect);
  }
  LOG_FATAL_IF(onLine(intersect, l) == false) << "intersect point is not on line";
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

double GeomCalc::lineDist(Line& l1, Line& l2, PtPair& closest)
{
  double dist, min_dist = std::numeric_limits<double>::max();
  Pt intersect;
  Side<Side<Pt>> pt;
  pt[kLeft] = l1;
  pt[kRight] = l2;
  size_t n1 = 2;
  size_t n2 = 2;
  if (isSame(l1[kHead], l2[kHead])) {
    n1 = 1;
  }
  if (isSame(l2[kHead], l1[kTail])) {
    n2 = 1;
  }
  if (n1 == 1 && n2 == 1) {
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

bool GeomCalc::isContain(const Trr& small, const Trr& huge)
{
  if (small.x_high() <= huge.x_high() + kEpsilon && small.x_low() >= huge.x_low() - kEpsilon && small.y_high() <= huge.y_high() + kEpsilon
      && small.y_low() >= huge.y_low() - kEpsilon) {
    return true;
  }
  return false;
}

void GeomCalc::buildTrr(Trr& ms, const double& r, Trr& build_trr)
{
  build_trr.x_low(ms.x_low() - r);
  build_trr.x_high(ms.x_high() + r);
  build_trr.y_low(ms.y_low() - r);
  build_trr.y_high(ms.y_high() + r);
}

void GeomCalc::lineToMs(Trr& ms, const Line& l)
{
  lineToMs(ms, l[kHead], l[kTail]);
}

void GeomCalc::lineToMs(Trr& ms, const Pt& p1, const Pt& p2)
{
  if (p1.y < p2.y) {
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