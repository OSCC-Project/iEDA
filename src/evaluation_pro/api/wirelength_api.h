#pragma once

#include "wirelength_db.h"

namespace ieval {

class WirelengthAPI
{
 public:
  WirelengthAPI();
  ~WirelengthAPI();

  TotalWLSummary totalWL(PointSets point_sets);
  NetWLSummary netWL(PointSet point_set);
  PathWLSummary pathWL(PointSet point_set, PointPair point_pair);

  int32_t totalEGRWL();
  int32_t netEGRWL(std::string net_name);
  int32_t pathEGRWL(std::string net_name, std::string point_name1, std::string point_name2);
};
}  // namespace ieval
