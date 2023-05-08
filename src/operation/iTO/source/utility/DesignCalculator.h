#pragma once

#include "Inst.h"
#include "Master.h"
#include "Rect.h"
#include "Utility.h"

namespace ito {
class Layout;

class DesignCalculator {
 public:
  DesignCalculator() = default;
  ~DesignCalculator() = default;
  // open functions
  void calculateDesign();

  static double calculateDesignArea(Layout *layout, int dbu);

  static double calculateCoreArea(ito::Rectangle core, int dbu);

  static double calcMasterArea(Master *master, int dbu);

 private:
  // init
  void initCore();
};

} // namespace ito