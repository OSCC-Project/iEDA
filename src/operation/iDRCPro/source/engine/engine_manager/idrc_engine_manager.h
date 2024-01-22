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

namespace idb {
class IdbLayer;
}  // namespace idb

namespace idrc {
/**
 *  DrcEngineManager definition : manage all shapes for all nets in all layers
 */
class DrcEngineManager
{
 public:
  DrcEngineManager(DrcDataManager* data_manager);
  ~DrcEngineManager();

  /// data manager
  std::map<idb::IdbLayer*, DrcEngineLayout*>& get_engine_layouts(LayoutType type = LayoutType::kRouting) { return _layouts[type]; }
  DrcEngineLayout* get_layout(idb::IdbLayer* layer, LayoutType type = LayoutType::kRouting);

  /// scanline manager
  std::map<LayoutType, std::map<idb::IdbLayer*, DrcEngineScanline*>>& get_scanline_matrix() { return _scanline_matrix; }
  std::map<idb::IdbLayer*, DrcEngineScanline*>& get_engine_scanlines(LayoutType type = LayoutType::kRouting)
  {
    return _scanline_matrix[type];
  }
  DrcEngineScanline* get_engine_scanline(idb::IdbLayer* layer, LayoutType type = LayoutType::kRouting);

  /// operator
  bool addRect(int llx, int lly, int urx, int ury, idb::IdbLayer* layer, int net_id = 0, LayoutType type = LayoutType::kRouting);

 private:
  DrcDataManager* _data_manager;
  /**
   * @definition
   *  _layouts : describe all shapes for all nets in all layers
   * @param
   *  LayoutType : cut | routing
   *  std::map<idb::IdbLayer*, DrcEngineLayout*> - idb::IdbLayer* : layer
   *  std::map<idb::IdbLayer*, DrcEngineLayout*> - DrcEngineLayout : describe all shapes for all nets in all layers
   */
  std::map<LayoutType, std::map<idb::IdbLayer*, DrcEngineLayout*>> _layouts;
  /**
   *  _scanline : scanline matrix
   */
  std::map<LayoutType, std::map<idb::IdbLayer*, DrcEngineScanline*>> _scanline_matrix;
};

}  // namespace idrc