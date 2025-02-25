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
#include <vector>

#include "condition_manager.h"
#include "engine_geometry_creator.h"
#include "engine_layout.h"
#include "engine_scanline.h"
#include "geometry_boost.h"
#include "idm.h"
#include "idrc_engine_manager.h"
#include "idrc_violation_manager.h"
#include "rule_condition_width.h"
#include "tech_rules.h"

namespace idrc {

DrcEngineManager::DrcEngineManager(DrcDataManager* data_manager, DrcConditionManager* condition_manager)
    : _data_manager(data_manager), _condition_manager(condition_manager)
{
  std::set<std::string> routing_layers;
  std::set<std::string> cut_layers;
  _layers.insert(std::make_pair(LayoutType::kRouting, routing_layers));
  _layers.insert(std::make_pair(LayoutType::kCut, cut_layers));

  std::map<std::string, DrcEngineLayout*> routing_map;
  std::map<std::string, DrcEngineLayout*> cut_map;
  _layouts.insert(std::make_pair(LayoutType::kRouting, routing_map));
  _layouts.insert(std::make_pair(LayoutType::kCut, cut_map));

  std::map<std::string, DrcEngineScanline*> routing_matrix_map;
  std::map<std::string, DrcEngineScanline*> cut_matrix_map;
  _scanline_matrix.insert(std::make_pair(LayoutType::kRouting, routing_matrix_map));
  _scanline_matrix.insert(std::make_pair(LayoutType::kCut, cut_matrix_map));
}

DrcEngineManager::~DrcEngineManager()
{
  for (auto& [type, layers] : _layers) {
    layers.clear();
  }
  _layers.clear();

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

std::map<std::string, DrcEngineLayout*>& DrcEngineManager::get_engine_layouts(LayoutType type)
{
  auto it = _layouts.find(type);
  if (it != _layouts.end()) {
    return it->second;
  } else {
    auto layout = std::make_pair(type, std::map<std::string, DrcEngineLayout*>{});
    _layouts.insert(layout);

    return layout.second;
  }
}

// get or create layout engine for each layer
DrcEngineLayout* DrcEngineManager::get_layout(std::string layer, LayoutType type)
{
  auto& layouts = get_engine_layouts(type);

  auto it = layouts.find(layer);
  if (it != layouts.end()) {
    return it->second;
  } else {
    DrcEngineLayout* engine_layout = new DrcEngineLayout(layer);
    layouts.insert(std::make_pair(layer, engine_layout));

    return engine_layout;
  }
}

std::set<std::string>& DrcEngineManager::get_layers(LayoutType type)
{
  auto it = _layers.find(type);
  if (it != _layers.end()) {
    return it->second;
  } else {
    auto layers = std::make_pair(type, std::set<std::string>{});
    _layers.insert(layers);

    return _layers.find(type)->second;
  }
}

bool DrcEngineManager::needChecking(std::string layer, LayoutType type)
{
  auto& layers = get_layers(type);

  return layers.find(layer) != layers.end() ? true : false;
}

void DrcEngineManager::addLayer(std::string layer, LayoutType type)
{
  auto& layers = get_layers(type);
  layers.insert(layer);
}

// add rect to engine
bool DrcEngineManager::addRect(int llx, int lly, int urx, int ury, std::string layer, int net_id, LayoutType type)
{
  /// get layout by type & layer id
  auto engine_layout = get_layout(layer, type);
  if (engine_layout == nullptr) {
    return false;
  }
  if (net_id >= 0 || net_id == NET_ID_VDD || net_id == NET_ID_VSS) {
    addLayer(layer, type);
  }

  return engine_layout->addRect(llx, lly, urx, ury, net_id);
}

void DrcEngineManager::dataPreprocess()
{
  std::vector<DrcEngineLayout*> layout_list;
  for (auto& [layer, layout] : get_engine_layouts(LayoutType::kRouting)) {
    layout_list.push_back(layout);
  }
#pragma omp parallel for num_threads(std::min((int)layout_list.size(),8))
  for (size_t i = 0; i < layout_list.size(); i++) {
    DrcEngineLayout* layout = layout_list[i];
    layout->combineLayout();
  }
}

void DrcEngineManager::filterData()
{
  for (auto& [layer, layout] : get_engine_layouts(LayoutType::kRouting)) {
    if (false == needChecking(layer, LayoutType::kRouting)) {
      continue;
    }

    DEBUGOUTPUT("Need to check layer:\t" << layer);

    /// area
    _condition_manager->checkArea(layer, layout);

    // overlap
    _condition_manager->checkOverlap(layer, layout);

    // min spacing
    _condition_manager->checkMinSpacing(layer, layout);

    // jog and prl
    _condition_manager->checkParallelLengthSpacing(layer, layout);
    _condition_manager->checkJogToJogSpacing(layer, layout);

    // edge
    _condition_manager->checkPolygons(layer, layout);
  }

  for (auto& [layer, layout] : get_engine_layouts(LayoutType::kCut)) {
    // TODO: cut rule
  }

  DEBUGOUTPUT("Finish drc checking:\t");
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