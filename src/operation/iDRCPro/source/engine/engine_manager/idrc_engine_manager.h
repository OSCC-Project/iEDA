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
#pragma once

#include <map>

#include "engine_layout.h"
#include "engine_scanline.h"
#include "idrc_data.h"

namespace idrc {
/**
 *  DrcEngineManager definition : manage all shapes for all nets in all layers
 */
class DrcEngineManager
{
 public:
  DrcEngineManager();
  ~DrcEngineManager();

  /// data manager
  std::map<int, DrcEngineLayout*>& get_engine_layouts(LayoutType type = LayoutType::kRouting) { return _layouts[type]; }
  DrcEngineLayout* get_layout(int layer_id, LayoutType type = LayoutType::kRouting);

  /// scanline manager
  std::map<LayoutType, std::map<int, DrcEngineScanline*>>& get_scanline_matrix(){return _scanline_matrix;}
  std::map<int, DrcEngineScanline*>& get_engine_scanlines(LayoutType type = LayoutType::kRouting) { return _scanline_matrix[type]; }
  DrcEngineScanline* get_engine_scanline(int layer_id, LayoutType type = LayoutType::kRouting);

  /// operator
  bool addRect(int llx, int lly, int urx, int ury, int layer_id = 0, int net_id = 0, LayoutType type = LayoutType::kRouting);

 private:
  /**
   * @definition
   *  _layouts : describe all shapes for all nets in all layers
   * @param
   *  LayoutType : cut | routing
   *  std::map<int, DrcEngineLayout*> - int : layer id
   *  std::map<int, DrcEngineLayout*> - DrcEngineLayout : describe all shapes for all nets in all layers
   */
  std::map<LayoutType, std::map<int, DrcEngineLayout*>> _layouts;
  /**
   *  _scanline : scanline matrix
   */
  std::map<LayoutType, std::map<int, DrcEngineScanline*>> _scanline_matrix;
};

}  // namespace idrc