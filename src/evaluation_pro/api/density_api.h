#pragma once

#include "density_db.h"

namespace ieval {
class DensityAPI
{
 public:
  DensityAPI();
  ~DensityAPI();

  DensityMapPathSummary getMapPathSummary();
};

}  // namespace ieval