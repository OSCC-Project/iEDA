#pragma once

#include "ScaleGrid.hpp"

namespace irt {

class GCellGrid : public ScaleGrid
{
 public:
  GCellGrid() = default;
  ~GCellGrid() = default;
  // getter

  // setter

  // function

 private:
};

struct CmpGCellGridASC
{
  bool operator()(GCellGrid& a, GCellGrid& b) const { return CmpScaleGridASC()(a, b); }
};

}  // namespace irt
