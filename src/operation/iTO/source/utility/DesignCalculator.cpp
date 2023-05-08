#include "DesignCalculator.h"

#include "Layout.h"

namespace ito {
double DesignCalculator::calculateDesignArea(Layout *layout, int dbu) {
  double design_area = 0.0;
  for (auto inst : layout->get_insts()) {
    Master *master = inst->get_master();
    design_area += calcMasterArea(master, dbu);
  }
  return design_area;
}

double DesignCalculator::calculateCoreArea(Rectangle core, int dbu) {
  double core_x = dbuToMeters(core.get_dx(), dbu);
  double core_y = dbuToMeters(core.get_dy(), dbu);
  return core_x * core_y;
}

double DesignCalculator::calcMasterArea(Master *master, int dbu) {
  //   if (!master->isAutoPlaceable()) {
  //     return 0;
  //   }
  double width = dbuToMeters(master->get_width(), dbu);
  double height = dbuToMeters(master->get_height(), dbu);
  return width * height;
}
} // namespace ito
