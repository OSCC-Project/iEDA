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
#include "idrc_engine_manager.h"

namespace idrc {

DrcEngineManager::DrcEngineManager()
{
  _layouts = {{LayoutType::kRouting, std::map<idb::IdbLayer*, DrcEngineLayout*>{}},
              {LayoutType::kCut, std::map<idb::IdbLayer*, DrcEngineLayout*>{}}};
  _scanline_matrix = {{LayoutType::kRouting, std::map<idb::IdbLayer*, DrcEngineScanline*>{}},
                      {LayoutType::kCut, std::map<idb::IdbLayer*, DrcEngineScanline*>{}}};
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
DrcEngineLayout* DrcEngineManager::get_layout(idb::IdbLayer* layer, LayoutType type)
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
bool DrcEngineManager::addRect(int llx, int lly, int urx, int ury, idb::IdbLayer* layer, int net_id, LayoutType type)
{
  /// get layout by type & layer id
  auto engine_layout = get_layout(layer, type);
  if (engine_layout == nullptr) {
    return false;
  }

  return engine_layout->addRect(llx, lly, urx, ury, net_id);
}

// get or create scanline engine for each layer
DrcEngineScanline* DrcEngineManager::get_engine_scanline(idb::IdbLayer* layer, LayoutType type)
{
  auto& scanline_engines = get_engine_scanlines(type);

  auto* scanline_engine = scanline_engines[layer];
  if (scanline_engine == nullptr) {
    scanline_engine = new DrcEngineScanline(layer);
    scanline_engines[layer] = scanline_engine;
  }

  return scanline_engine;
}

}  // namespace idrc