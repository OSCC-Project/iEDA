/*
 * @FilePath: wirelength_db.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

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
  int64_t HPWL;
  int64_t FLUTE;
  int64_t HTree;
  int64_t VTree;
  int64_t GRWL;
};

struct NetWLSummary
{
  int32_t HPWL;
  int32_t FLUTE;
  int32_t HTree;
  int32_t VTree;
  int32_t GRWL;
};

struct PathWLSummary
{
  int32_t HPWL;
  int32_t FLUTE;
  int32_t HTree;
  int32_t VTree;
};

}  // namespace ieval