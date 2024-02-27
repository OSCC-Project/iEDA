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
#include "idrc_engine_manager.h"

#include "engine_geometry_creator.h"
#include "engine_scanline.h"
#include "geometry_boost.h"
#include "idm.h"
#include "idrc_engine_manager.h"
#include "omp.h"
#include "rule_condition_width.h"
#include "tech_rules.h"

namespace idrc {

DrcEngineManager::DrcEngineManager(DrcDataManager* data_manager, DrcConditionManager* condition_manager)
    : _data_manager(data_manager), _condition_manager(condition_manager)
{
  _layouts
      = {{LayoutType::kRouting, std::map<std::string, DrcEngineLayout*>{}}, {LayoutType::kCut, std::map<std::string, DrcEngineLayout*>{}}};
  _scanline_matrix = {{LayoutType::kRouting, std::map<std::string, DrcEngineScanline*>{}},
                      {LayoutType::kCut, std::map<std::string, DrcEngineScanline*>{}}};
  // _engine_check = new DrcEngineCheck();
}

DrcEngineManager::~DrcEngineManager()
{
  for (auto& [type, layout_arrays] : _layouts) {
    for (auto& [layer, layout] : layout_arrays) {
      if (layout != nullptr) {
        delete layout;
        layout = nullptr;
      }
    }

    layout_arrays.clear();
  }
  _layouts.clear();

  for (auto& [type, matrix_arrays] : _scanline_matrix) {
    for (auto& [layer, matrix] : matrix_arrays) {
      if (matrix != nullptr) {
        delete matrix;
        matrix = nullptr;
      }
    }
    matrix_arrays.clear();
  }
  _scanline_matrix.clear();
}

// get or create layout engine for each layer
DrcEngineLayout* DrcEngineManager::get_layout(std::string layer, LayoutType type)
{
  auto& layouts = get_engine_layouts(type);

  auto* engine_layout = layouts[layer];
  if (engine_layout == nullptr) {
    engine_layout = new DrcEngineLayout(layer);
    layouts[layer] = engine_layout;
  }

  return engine_layout;
}
// add rect to engine
bool DrcEngineManager::addRect(int llx, int lly, int urx, int ury, std::string layer, int net_id, LayoutType type)
{
  /// get layout by type & layer id
  auto engine_layout = get_layout(layer, type);
  if (engine_layout == nullptr) {
    return false;
  }

  return engine_layout->addRect(llx, lly, urx, ury, net_id);
}

void DrcEngineManager::combineLayouts()
{
  for (auto& [layer, layout] : get_engine_layouts()) {
    layout->combineLayout();
  }
}

void DrcEngineManager::dataPreprocess()
{
  combineLayouts();
}

void DrcEngineManager::filterData()
{
  dataPreprocess();

  // TODO: put logic bellow into condition module
  // TODO: multi-thread
  for (auto& [layer, layout] : get_engine_layouts()) {
    // TODO: only for routing layers
    auto& layer_polyset = layout->get_layout()->get_engine()->get_polyset();
#ifdef DEBUG_IDRC_ENGINE
    std::vector<ieda_solver::GeometryViewPolygon> polygons;
    layer_polyset.get(polygons);
#endif

    // overlap
    auto& overlap = layout->get_layout()->get_engine()->getOverlap();
    if (overlap.size() > 0) {
      // TODO: create scanline point data
      int a = 0;
    }

    // min spacing
    int min_spacing = DrcTechRuleInst->getMinSpacing(layer);
    if (min_spacing > 0) {
      auto set = layer_polyset;
      gtl::grow_and(set, min_spacing / 2);

#ifdef DEBUG_IDRC_ENGINE
      std::vector<ieda_solver::GeometryViewPolygon> grow_polygons;
      set.get(grow_polygons);
      if (grow_polygons.size() > 0) {
        int a = 0;
      }
#endif
    }

    // jog and prl
    using WidthToPolygonSetMap = std::map<int, ieda_solver::GeometryPolygonSet>;
    WidthToPolygonSetMap jog_cut_rect_map_horizontal;
    WidthToPolygonSetMap jog_cut_rect_map_vertical;
    WidthToPolygonSetMap jog_cut_region_map_horizontal;
    WidthToPolygonSetMap jog_cut_region_map_vertical;
    WidthToPolygonSetMap jog_wire_map_horizontal;
    WidthToPolygonSetMap jog_wire_map_vertical;

    WidthToPolygonSetMap prl_wire_map;

    auto& wires = layout->get_layout()->get_engine()->getWires();
    for (auto& wire : wires) {
      // jog
      auto wire_direction = ieda_solver::getWireDirection(wire);
      auto width_direction = wire_direction.get_perpendicular();
      int wire_width = ieda_solver::getWireWidth(wire, width_direction);

      auto& jog_cut_rect_map = wire_direction == ieda_solver::HORIZONTAL ? jog_cut_rect_map_horizontal : jog_cut_rect_map_vertical;
      auto& jog_cut_region_map = wire_direction == ieda_solver::HORIZONTAL ? jog_cut_region_map_horizontal : jog_cut_region_map_vertical;
      auto& jog_wire_map = wire_direction == ieda_solver::HORIZONTAL ? jog_wire_map_horizontal : jog_wire_map_vertical;

      auto rule_jog_to_jog = DrcTechRuleInst->getJogToJog(layer);
      if (rule_jog_to_jog) {
        for (auto& width_item : rule_jog_to_jog->get_width_list()) {
          int rule_width = width_item.get_width();
          if (wire_width > rule_width) {
            // create big wire layer
            jog_wire_map[rule_width] += wire;
            // create within rects layer
            auto expand_rects = ieda_solver::getExpandRects(wire, width_item.get_par_within(), width_direction);
            for (auto& rect : expand_rects) {
              jog_cut_rect_map[rule_width] += rect;
              jog_cut_region_map[rule_width] += rect & layer_polyset;
            }
            break;
          }
        }
      }
    }

    // jog
    auto deal_with_jog = [&](ieda_solver::GeometryOrientation wire_direction, WidthToPolygonSetMap& jog_wire_map,
                             WidthToPolygonSetMap& jog_cut_rect_map, WidthToPolygonSetMap& jog_cut_region_map) {
      for (auto& [rule_width, jog_wires] : jog_wire_map) {
        auto jogs_attach_wires = jog_cut_region_map[rule_width];
        ieda_solver::interact(jogs_attach_wires, jog_wires);
        auto wire_with_jogs = jog_wire_map[rule_width] + jogs_attach_wires;

        auto jogs_opposite = jog_cut_region_map[rule_width] - jogs_attach_wires;

        auto region_B = jog_cut_rect_map[rule_width] - jog_cut_region_map[rule_width];
        std::vector<ieda_solver::GeometryRect> regions;
        ieda_solver::getRectangles(regions, region_B, wire_direction.get_perpendicular());

        ieda_solver::GeometryRect envelope_cut_rects;
        ieda_solver::envelope(envelope_cut_rects, jog_cut_rect_map[rule_width]);
        auto cut_rects_negative_regions = envelope_cut_rects - jog_cut_rect_map[rule_width];

        for (auto& rect_region : regions) {
          // TODO: get region a
        }

        int a = 0;
      }
    };
    if (!jog_wire_map_horizontal.empty()) {
      deal_with_jog(ieda_solver::HORIZONTAL, jog_wire_map_horizontal, jog_cut_rect_map_horizontal, jog_cut_region_map_horizontal);
    }
    if (!jog_wire_map_vertical.empty()) {
      deal_with_jog(ieda_solver::VERTICAL, jog_wire_map_vertical, jog_cut_rect_map_vertical, jog_cut_region_map_vertical);
    }

    // TODO: rule check
  }
}

// void DrcEngineManager::dataPreprocess()
// {
// #ifdef DEBUG_IDRC_ENGINE
//   ieda::Stats stats;
//   std::cout << "idrc : begin init scanline database" << std::endl;
// #endif
//   /// init scanline engine for routing layer
//   auto& layouts = get_engine_layouts(LayoutType::kRouting);

//   for (auto& [layer, engine_layout] : layouts) {
//     /// scanline engine for one layer
//     auto* scanline_engine = get_engine_scanline(layer, LayoutType::kRouting);
//     auto* scanline_preprocess = scanline_engine->get_preprocess();

//     // reserve capacity for basic points
//     uint64_t point_number = engine_layout->pointCount();
//     scanline_preprocess->reserveSpace(point_number);

//     // create scanline points
//     for (auto [net_id, sub_layout] : engine_layout->get_sub_layouts()) {
//       /// build engine data
//       auto* boost_engine = static_cast<ieda_solver::GeometryBoost*>(sub_layout->get_engine());
//       auto boost_pt_list_pair = boost_engine->get_boost_polygons_points();

//       /// add data to scanline engine
//       scanline_preprocess->addData(boost_pt_list_pair.second, net_id);  /// boost_pt_list_pair : second value is polygon points
//     }

//     /// sort point list in scanline data manager
//     scanline_preprocess->sortEndpoints();

//     // std::cout << "idrc : layer id = " << layer->get_id() << " polygon points total number = " << point_number << std::endl;
//   }

// #ifdef DEBUG_IDRC_ENGINE
//   std::cout << "idrc : end init scanline database, "
//             << " runtime = " << stats.elapsedRunTime() << " memory = " << stats.memoryDelta() << std::endl;
// #endif
// }

// void DrcEngineManager::filterData()
// {
//   dataPreprocess();

// #ifdef DEBUG_IDRC_ENGINE
//   ieda::Stats stats;

//   std::cout << "idrc : begin scanline" << std::endl;
// #endif
//   /// run scanline method for all routing layers
//   auto& layouts = get_engine_layouts(LayoutType::kRouting);
//   for (auto& [layer, engine_layout] : layouts) {
//     /// scanline engine for each layer
//     auto* scanline_engine = get_engine_scanline(layer, LayoutType::kRouting);
//     scanline_engine->doScanline();
//   }

// #ifdef DEBUG_IDRC_ENGINE
//   std::cout << "idrc : end scanline, "
//             << " runtime = " << stats.elapsedRunTime() << " memory = " << stats.memoryDelta() << std::endl;
// #endif
// }

// get or create scanline engine for each layer
DrcEngineScanline* DrcEngineManager::get_engine_scanline(std::string layer, LayoutType type)
{
  auto& scanline_engines = get_engine_scanlines(type);

  auto* scanline_engine = scanline_engines[layer];
  if (scanline_engine == nullptr) {
    scanline_engine = new DrcEngineScanline(layer, this, _condition_manager);
    scanline_engines[layer] = scanline_engine;
  }

  return scanline_engine;
}

}  // namespace idrc