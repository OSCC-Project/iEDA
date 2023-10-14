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
 * @file GeomCalc.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include "Components.hh"

namespace icts {
namespace bst {
enum class LineType
{
  kVertical,
  kHorizontal,
  kManhattan,
  kFlat,
  kTilt,
};

enum class IntersectType
{
  kNone,
  kEndpoint,  // intersect point is endpoint of line segment
  kCrossing,  // intersect point is in both line segment
  kOverlap,   // intersect some part of line segment
  kSame,      // two line are same
};

enum class RelativeType
{
  kLeft,
  kRight,
  kTop,
  kBottom,
  kManhattanParallel,
};

class GeomCalc
{
 public:
  GeomCalc() = delete;
  /* Calculate */
  // point
  static bool isSame(const Pt& p1, const Pt& p2);

  static double distance(const Pt& p1, const Pt& p2);

  static double ptToLineDistManhattan(Pt& p, const Line& l, Pt& closest);

  static double ptToLineDistNotManhattan(Pt& p, const Line& l, Pt& closest);

  static double ptToLineDist(Pt& p, const Line& l, Pt& closest);

  static double ptToTrrDist(Pt& p, Trr& ms);

  static void calcCoord(Pt& p, const Line& l, const double& shift);

  static void calcRelativeCoord(Pt& p, const RelativeType& type, const double& shift);

  static double crossProduct(const Pt& p1, const Pt& p2, const Pt& p3);
  // line
  static LineType lineType(const Line& l);

  static LineType lineType(const Pt& p1, const Pt& p2);

  static IntersectType lineIntersect(Pt& p, Line& l1, Line& l2);

  static RelativeType lineRelative(const Line& l1, const Line& l2, const size_t& ref);

  static double lineDist(Line& l1, Line& l2, PtPair& closest);

  static bool onLine(Pt& p, const Line& l);

  static bool isParallel(const Line& l1, const Line& l2);
  // box
  static bool inBoundBox(const Pt& p, const Line& l);

  static bool boundBoxOverlap(const Line& l1, const Line& l2, const double& epsilon = kEpsilon);

  static bool boundBoxOverlap(const double& x1, const double& y1, const double& x2, const double& y2, const double& x3, const double& y3,
                              const double& x4, const double& y4, const double& epsilon = kEpsilon);

  // Trr
  static double msDistance(Trr& ms1, Trr& ms2);
  static void makeIntersect(Trr& ms1, Trr& ms2, Trr& intersect);
  static void coreMidPoint(Trr& ms, Pt& mid);
  static bool isTrrContain(const Trr& small, const Trr& large);
  static void buildTrr(const Trr& ms, const double& r, Trr& build_trr);
  static void trrCore(const Trr& trr, Trr& core);
  static void trrToPt(const Trr& trr, Pt& pt);
  static void trrToRegion(Trr& trr, Region& region);
  static bool isSegmentTrr(const Trr& trr);
  // Pts
  static void sortPtsByFront(Pts& pts);
  static void sortPtsByVal(Pts& pts);
  static void sortPtsByValDec(Pts& pts);
  static void uniquePtsLoc(std::vector<Pt>& pts);
  static void uniquePtsVal(std::vector<Pt>& pts);
  // Region
  static std::vector<Line> getLines(const std::vector<Pt>& pts);
  static void convexHull(std::vector<Pt>& pts);
  static Pt centerPt(const std::vector<Pt>& pts);
  static bool isRegionContain(const Pt& p, const std::vector<Pt>& region);
  static Pt closestPtOnRegion(const Pt& p, const std::vector<Pt>& region);
  /* Convert */
  static void lineToMs(Trr& ms, const Line& l);
  static void lineToMs(Trr& ms, const Pt& p1, const Pt& p2);
  static void msToLine(Trr& ms, Line& l);
  static void msToLine(Trr& ms, Pt& p1, Pt& p2);
  /* Check */
  static void checkMs(Trr& ms);
};

}  // namespace bst
}  // namespace icts