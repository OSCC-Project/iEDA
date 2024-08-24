#pragma once

#include "congestion_db.h"

namespace ieval {

class CongestionAPI
{
 public:
  CongestionAPI();
  ~CongestionAPI();

  OverflowSummary getOverflowSummary();
  MapPathSummary getMapPathSummary();
};

}  // namespace ieval