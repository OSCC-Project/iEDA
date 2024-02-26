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
#include "idrc_engine.h"

#include "engine_init_def.h"
#include "engine_init_rt.h"
#include "idm.h"

namespace idrc {

DrcEngine::DrcEngine(DrcDataManager* data_manager, DrcConditionManager* condition_manager)
{
  _data_manager = data_manager;
  _condition_manager = condition_manager;
  _engine_manager = new DrcEngineManager(data_manager, condition_manager);
}

DrcEngine::~DrcEngine()
{
  if (_engine_manager != nullptr) {
    delete _engine_manager;
    _engine_manager = nullptr;
  }
}

void DrcEngine::initEngine(DrcCheckerType checker_type)
{
  switch (checker_type) {
    case DrcCheckerType::kDef: {
#ifdef DEBUG_IDRC_ENGINE
      std::cout << "idrc : start init engine def" << std::endl;
      ieda::Stats stats_def;
#endif
      initEngineDef();
#ifdef DEBUG_IDRC_ENGINE
      std::cout << "idrc : init engine def time = " << stats_def.elapsedRunTime() << " memory = " << stats_def.memoryDelta() << std::endl;
#endif

      break;
    }
    case DrcCheckerType::kRT:
      initEngineGeometryData();

    default:
      break;
  }
  _engine_manager->combineLayouts();
}

/**
 * init engine data from RT data
 */
void DrcEngine::initEngineGeometryData()
{
  DrcEngineInitRT init_rt(_engine_manager, _data_manager);
  init_rt.init();
}
/**
 * init engine data from idb
 */
void DrcEngine::initEngineDef()
{
  DrcEngineInitDef init_def(_engine_manager);
  init_def.init();
}

void DrcEngine::operateEngine()
{
  _engine_manager->filterData();
}

void DrcEngine::checkEngine()
{
  // _engine_manager->get_engine_check()->check();
}

}  // namespace idrc