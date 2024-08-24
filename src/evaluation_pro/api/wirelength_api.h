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
  PathWLSummary pathWL(PointSet point_set);
};
}  // namespace ieval
