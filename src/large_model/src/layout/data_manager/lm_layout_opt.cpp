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

#include "lm_layout_opt.h"

#include "IdbGeometry.h"
#include "IdbLayer.h"
#include "IdbLayerShape.h"
#include "IdbNet.h"
#include "IdbRegularWire.h"
#include "IdbSpecialNet.h"
#include "IdbSpecialWire.h"
#include "Log.hh"
#include "idm.h"
#include "omp.h"
#include "usage.hh"

namespace ilm {

void LmLayoutOptimize::wirePruning()
{
  ieda::Stats stats;

  LOG_INFO << "LM optimize connections for routing layer start...";

  omp_lock_t lck;
  omp_init_lock(&lck);

  auto& net_map = _layout->get_graph().get_net_map();

  struct ClassfyMap
  {
    int pin_num;
    std::vector<std::vector<LmNetWire*>> wire_map;
  };

  for (auto& [net_id, net] : net_map) {
    auto& pin_ids = net.get_pin_ids();

    ClassfyMap classify_map;
    for (int i = 0; i < net.get_wires().size(); i++) {
    }
  }

  omp_destroy_lock(&lck);

  LOG_INFO << "LM optimize connections for routing layer end...";
}

}  // namespace ilm