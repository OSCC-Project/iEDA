#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ieval {

using Point = std::pair<int32_t, int32_t>;
using PointPair = std::pair<Point, Point>;
using PointSet = std::vector<Point>;
using PointSets = std::vector<PointSet>;

struct TotalWLSummary
{
  int32_t HPWL;
  int32_t FLUTE;
  int32_t HTree;
  int32_t VTree;
};

struct NetWLSummary
{
  int32_t HPWL;
  int32_t FLUTE;
  int32_t HTree;
  int32_t VTree;
};

struct PathWLSummary
{
  int32_t HPWL;
  int32_t FLUTE;
  int32_t HTree;
  int32_t VTree;
};

}  // namespace ieval