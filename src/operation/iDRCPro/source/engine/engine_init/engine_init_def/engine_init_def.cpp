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

#include "engine_init_def.h"

#include "IdbGeometry.h"
#include "IdbLayer.h"
#include "IdbLayerShape.h"
#include "IdbNet.h"
#include "IdbRegularWire.h"
#include "IdbSpecialNet.h"
#include "IdbSpecialWire.h"
#include "idm.h"
#include "idrc_data.h"
#include "idrc_engine_manager.h"

namespace idrc {
/**
 *  top flow to init all def data to geometry data
 */
void DrcEngineInitDef::init()
{
  initDataFromIOPins();
  initDataFromInstances();
  initDataFromPDN();
  initDataFromNets();
}

void DrcEngineInitDef::initDataFromIOPins()
{
  auto* idb_design = dmInst->get_idb_design();
  auto* idb_io_pins = idb_design->get_io_pin_list();

  for (auto* idb_io_pin : idb_io_pins->get_pin_list()) {
    if (idb_io_pin != nullptr && idb_io_pin->get_term()->is_placed()) {
      initDataFromPin(idb_io_pin);
    }
  }
}

void DrcEngineInitDef::initDataFromInstances()
{
#ifdef DEBUG_IDRC_ENGINE_INIT
  std::cout << "idrc : begin init data from instances" << std::endl;
  ieda::Stats stats;
#endif

  auto* idb_design = dmInst->get_idb_design();

  uint64_t number = 0;
  for (IdbInstance* idb_inst : idb_design->get_instance_list()->get_instance_list()) {
    if (idb_inst == nullptr || idb_inst->get_cell_master() == nullptr) {
      continue;
    }
    /// obs
    for (auto* idb_pin : idb_inst->get_pin_list()->get_pin_list()) {
      initDataFromPin(idb_pin);
    }
    for (auto* idb_obs : idb_inst->get_obs_box_list()) {
      initDataFromShape(idb_obs, NET_ID_OBS);
    }

    number++;
  }

#ifdef DEBUG_IDRC_ENGINE_INIT
  std::cout << "idrc : end init data from instances, instance number = " << number << " runtime = " << stats.elapsedRunTime()
            << " memory = " << stats.memoryDelta() << std::endl;
#endif
}

void DrcEngineInitDef::initDataFromPDN()
{
#ifdef DEBUG_IDRC_ENGINE_INIT
  std::cout << "idrc : begin init data from pdn" << std::endl;
  ieda::Stats stats;
#endif

  auto* idb_design = dmInst->get_idb_design();

  uint64_t number = 0;
  for (auto* idb_special_net : idb_design->get_special_net_list()->get_net_list()) {
    int net_id = idb_special_net->is_vdd() ? NET_ID_VDD : NET_ID_VSS;
    for (auto* idb_special_wire : idb_special_net->get_wire_list()->get_wire_list()) {
      for (auto* idb_segment : idb_special_wire->get_segment_list()) {
        /// add wire
        if (idb_segment->get_point_list().size() >= 2) {
          /// get routing width
          auto* routing_layer = dynamic_cast<IdbLayerRouting*>(idb_segment->get_layer());
          int32_t routing_width = idb_segment->get_route_width() == 0 ? routing_layer->get_width() : idb_segment->get_route_width();

          /// calculate rectangle by two points
          auto* point_1 = idb_segment->get_point_start();
          auto* point_2 = idb_segment->get_point_second();

          initDataFromPoints(point_1, point_2, routing_width, idb_segment->get_layer(), net_id, true);
        }

        /// vias
        if (idb_segment->is_via()) {
          /// add via
          initDataFromVia(idb_segment->get_via(), net_id);
        }

        number++;
      }
    }
  }

#ifdef DEBUG_IDRC_ENGINE_INIT
  std::cout << "idrc : end init data from pdn, segment number = " << number << " runtime = " << stats.elapsedRunTime()
            << " memory = " << stats.memoryDelta() << std::endl;
#endif
}

/**
 * the basic geometry unit is construct independently by layer id and net id,
 * so it enable to read net parallelly
 */
void DrcEngineInitDef::initDataFromNets()
{
#ifdef DEBUG_IDRC_ENGINE_INIT
  std::cout << "idrc : begin init data from nets" << std::endl;

  ieda::Stats stats;
#endif

  auto* idb_design = dmInst->get_idb_design();

  for (auto* idb_net : idb_design->get_net_list()->get_net_list()) {
    initDataFromNet(idb_net);
  }

#ifdef DEBUG_IDRC_ENGINE_INIT
  std::cout << "idrc : end init data from nets, net number = " << idb_design->get_net_list()->get_num()
            << " runtime = " << stats.elapsedRunTime() << " memory = " << stats.memoryDelta() << std::endl;
#endif
}

}  // namespace idrc