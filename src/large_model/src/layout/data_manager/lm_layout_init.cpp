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

#include "lm_layout_init.h"

#include "IdbGeometry.h"
#include "IdbLayer.h"
#include "IdbLayerShape.h"
#include "IdbNet.h"
#include "IdbRegularWire.h"
#include "IdbSpecialNet.h"
#include "IdbSpecialWire.h"
#include "Log.hh"
#include "idm.h"

namespace ilm {

void LmLayoutInit::init()
{
  initDie();
  initCore();
  initViaIds();
  initCellMasters();

  initLayers();
  initTracks();
  initPDN();
  initInstances();
  initNets();
}

void LmLayoutInit::initViaIds()
{
  auto* idb_layout = dmInst->get_idb_layout();
  auto* idb_design = dmInst->get_idb_design();
  auto* idb_lef_vias = idb_layout->get_via_list();
  auto* idb_def_vias = idb_design->get_via_list();

  auto& via_map = _layout->get_via_id_map();

  int index = 0;
  for (auto* via : idb_lef_vias->get_via_list()) {
    via_map.insert(std::make_pair(via->get_name(), index++));
  }

  for (auto* via : idb_def_vias->get_via_list()) {
    via_map.insert(std::make_pair(via->get_name(), index++));
  }

  LOG_INFO << "Via number : " << index;
}

void LmLayoutInit::initCellMasters()
{
}

void LmLayoutInit::initDie()
{
  auto* idb_layout = dmInst->get_idb_layout();
  auto* idb_die = idb_layout->get_die();

  auto& patch_layers = _layout->get_patch_layers();
  for (auto& [order, patch_layer] : patch_layers.get_patch_layer_map()) {
    patch_layer.set_llx(idb_die->get_llx());
    patch_layer.set_lly(idb_die->get_lly());
    patch_layer.set_urx(idb_die->get_urx());
    patch_layer.set_ury(idb_die->get_ury());

    auto& grid = patch_layer.get_grid();
    grid.set_llx(idb_die->get_llx());
    grid.set_lly(idb_die->get_lly());
    grid.set_urx(idb_die->get_urx());
    grid.set_ury(idb_die->get_ury());
  }
}

void LmLayoutInit::initCore()
{
}

void LmLayoutInit::initLayers()
{
  auto* idb_layout = dmInst->get_idb_layout();
  auto* idb_layers = idb_layout->get_layers();
  auto idb_layer_1st = dmInst->get_config().get_routing_layer_1st();

  auto& layer_id_map = _layout->get_layer_id_map();
  auto& patch_layers = _layout->get_patch_layers();
  auto& patch_layer_map = patch_layers.get_patch_layer_map();

  bool b_record = false;
  int index = 0;
  for (auto* idb_layer : idb_layers->get_layers()) {
    if (idb_layer->get_name() == idb_layer_1st) {
      b_record = true;
    }

    if (true == b_record) {
      layer_id_map.insert(std::make_pair(idb_layer->get_name(), index++));

      LmPatchLayer patch_layer;
      patch_layer.set_layer_name(idb_layer->get_name());
      patch_layer.set_layer_order(index);

      auto& grid = patch_layer.get_grid();
      grid.set_layer_order(index);

      patch_layer_map.insert(std::make_pair(index, patch_layer));
    }
  }

  patch_layers.set_layer_order_bottom(0);
  patch_layers.set_layer_order_top(index - 1);

  LOG_INFO << "Layer number : " << index;
}

void LmLayoutInit::initTracks()
{
  auto* idb_layout = dmInst->get_idb_layout();
  auto* idb_layers = idb_layout->get_layers();

  int order = idb_layers->get_layer_order(dmInst->get_config().get_routing_layer_1st());
}

void LmLayoutInit::initPDN()
{
}

void LmLayoutInit::initInstances()
{
}

void LmLayoutInit::initIOPins()
{
}

void LmLayoutInit::initNets()
{
}

}  // namespace ilm