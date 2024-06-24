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

#include "EstimateParasitics.h"
#include "Placer.h"
#include "ToConfig.h"
#include "data_manager.h"
#include "timing_engine.h"
#include "timing_engine_util.h"

namespace ito {

bool ToTimingEngine::repowerInstance(Pin* driver_pin)
{
  TOLibRepowerInstance repower_inst(driver_pin);
  return repower_inst.repowerInstance();
}

bool ToTimingEngine::repowerInstance(ista::LibCell* repower_size, ista::Instance* repowered_inst)
{
  TOLibRepowerInstance repower_inst;
  return repower_inst.repowerInstance(repower_size, repowered_inst);
}

void ToTimingEngine::placeInstance(int x, int y, ista::Instance* place_inst)
{
  TimingIDBAdapter* idb_adapter = get_sta_adapter();
  idb::IdbInstance* idb_inst = idb_adapter->staToDb(place_inst);
  unsigned master_width = idb_inst->get_cell_master()->get_width();
  std::pair<int, int> loc = toPlacer->findNearestSpace(master_width, x, y);
  idb_inst->set_status_placed();
  idb_inst->set_coodinate(loc.first, loc.second);
  // set orient
  auto row = toPlacer->findRow(loc.second);
  if (row) {
    idb_inst->set_orient(row->get_site()->get_orient());
  } else {
    idb_inst->set_orient(idb::IdbOrient::kN_R0);
  }
  toPlacer->updateRow(master_width, loc.first, loc.second);
}

}  // namespace ito