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

#include "engine_init_scanline.h"

#include "engine_layout.h"
#include "engine_scanline.h"
#include "geometry_boost.h"
#include "idrc_engine_manager.h"
// #include "usage.hh"
#include "tech_rules.h"

namespace idrc {

void DrcEngineInitScanline::init()
{
  initGeometryData();
  initScanlineResult();
}
/**
 * init data from geometry data
 */
void DrcEngineInitScanline::initGeometryData()
{
  //   ieda::Stats stats;
  //   std::cout << "idrc : begin init scanline database" << std::endl;
  /// init scanline engine for routing layer
  auto& layouts = _engine_manager->get_engine_layouts(LayoutType::kRouting);

  for (auto& [layer, engine_layout] : layouts) {
    /// scanline engine for one layer
    auto* scanline_engine = _engine_manager->get_engine_scanline(layer, LayoutType::kRouting);
    auto* scanline_dm = scanline_engine->get_data_manager();

    // reserve capacity for basic points
    uint64_t point_number = engine_layout->pointCount();
    scanline_dm->reserveSpace(point_number);

    // create scanline points
    for (auto [net_id, sub_layout] : engine_layout->get_sub_layouts()) {
      /// build engine data
      auto* boost_engine = static_cast<ieda_solver::GeometryBoost*>(sub_layout->get_engine());
      auto boost_pt_list_pair = boost_engine->get_boost_polygons_points();

      /// add data to scanline engine
      scanline_dm->addData(boost_pt_list_pair.second, net_id);  /// boost_pt_list_pair : second value is polygon points
    }

    /// sort point list in scanline data manager
    scanline_dm->sortEndpoints();

    // std::cout << "idrc : layer id = " << layer->get_id() << " polygon points total number = " << point_number << std::endl;
  }

  //   std::cout << "idrc : end init scanline database, "
  //             << " runtime = " << stats.elapsedRunTime() << " memory = " << stats.memoryDelta() << std::endl;
}
/**
 * build scanline result data as the basic data
 */
void DrcEngineInitScanline::initScanlineResult()
{
  //   ieda::Stats stats;

  //   std::cout << "idrc : begin scanline" << std::endl;
  /// run scanline method for all routing layers
  auto& layouts = _engine_manager->get_engine_layouts(LayoutType::kRouting);
  for (auto& [layer, engine_layout] : layouts) {
    auto* rule_routing_layer = DrcTechRuleInst->get_rule_routing_layer(layer);
    if (rule_routing_layer == nullptr) {
      continue;
    }
    auto* rule_map = rule_routing_layer->get_condition_map(RuleType::kSpacing);
    int min_spacing = rule_map->get_min();
    int max_spacing = rule_map->get_max();

    /// scanline engine for each layer
    auto* scanline_engine = _engine_manager->get_engine_scanline(layer, LayoutType::kRouting);
    scanline_engine->doScanline(min_spacing, max_spacing);
  }

  //   std::cout << "idrc : end scanline, "
  //             << " runtime = " << stats.elapsedRunTime() << " memory = " << stats.memoryDelta() << std::endl;
}

}  // namespace idrc