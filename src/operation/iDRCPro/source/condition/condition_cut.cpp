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

#include <string>

#include "condition_manager.h"
#include "engine_geometry_creator.h"
#include "engine_layout.h"
#include "idm.h"

namespace idrc {

void DrcConditionManager::checkCutSpacing(std::string layer, DrcEngineLayout* layout)
{
}

void DrcConditionManager::checkCutArraySpacing(std::string layer, DrcEngineLayout* layout)
{
}

void DrcConditionManager::checkCutEnclosure(std::string layer, DrcEngineLayout* layout)
{
}

void DrcConditionManager::checkCutOverlap(std::string layer, DrcEngineLayout* layout)
{
  auto shrink_rect = [](ieda_solver::GeometryRect& rect, int value) -> bool {
    ieda_solver::GeometryRect result;
    int with = ieda_solver::getWireWidth(rect, ieda_solver::HORIZONTAL);
    int height = ieda_solver::getWireWidth(rect, ieda_solver::HORIZONTAL);
    if (with < 2 * value || height < 2 * value) {
      return false;
    }

    ieda_solver::shrink(rect, ieda_solver::HORIZONTAL, value);
    ieda_solver::shrink(rect, ieda_solver::VERTICAL, value);

    return true;
  };

  ieda::Stats states;

  ieda_solver::EngineGeometryCreator geo_creator;
  auto* engine = dynamic_cast<ieda_solver::GeometryBoost*>(geo_creator.create());

  for (auto& [net_id, sub_layout] : layout->get_sub_layouts()) {
    auto sub_polyset = sub_layout->get_engine()->copyPolyset();
    sub_polyset.clean();
    sub_polyset.bloat2(1, 1, 1, 1);
    engine->addPolyset(sub_polyset);
  }

  int total_drc = 0;
  auto overlaps = engine->getOverlap();
  for (auto& overlap : overlaps) {
    std::vector<ieda_solver::GeometryRect> results;
    ieda_solver::getDefaultRectangles(results, overlap);

    for (auto rect : results) {
      if (shrink_rect(rect, 1)) {
        addViolation(rect, layer, ViolationEnumType::kCutShort);
        total_drc++;
      }
    }
  }

  DEBUGOUTPUT(DEBUGHIGHLIGHT("Cut Short:\t") << total_drc << "\tlayer " << layer << "\tnets = " << layout->get_sub_layouts().size()
                                               << "\ttime = " << states.elapsedRunTime() << "\tmemory = " << states.memoryDelta());

  delete engine;
}

void DrcConditionManager::checkCutWidth(std::string layer, DrcEngineLayout* layout)
{
}

}  // namespace idrc