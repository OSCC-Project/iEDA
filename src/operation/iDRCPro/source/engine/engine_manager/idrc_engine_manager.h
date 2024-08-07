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

class DrcConditionManager;
/**
 *  DrcEngineManager definition : manage all shapes for all nets in all layers
 */
class DrcEngineManager
{
 public:
  DrcEngineManager(DrcDataManager* data_manager, DrcConditionManager* condition_manager);
  ~DrcEngineManager();

  /// data manager
  std::map<std::string, DrcEngineLayout*>& get_engine_layouts(LayoutType type = LayoutType::kRouting);
  DrcEngineLayout* get_layout(std::string layer, LayoutType type = LayoutType::kRouting);

  /// scanline manager
  std::map<LayoutType, std::map<std::string, DrcEngineScanline*>>& get_scanline_matrix() { return _scanline_matrix; }
  std::map<std::string, DrcEngineScanline*>& get_engine_scanlines(LayoutType type = LayoutType::kRouting) { return _scanline_matrix[type]; }
  DrcEngineScanline* get_engine_scanline(std::string layer, LayoutType type = LayoutType::kRouting);

  // DrcEngineCheck* get_engine_check() { return _engine_check; }

  /// operator
  bool addRect(int llx, int lly, int urx, int ury, std::string layer, int net_id = 0, LayoutType type = LayoutType::kRouting);

  void dataPreprocess();

  void filterData();

 private:
  DrcDataManager* _data_manager;
  DrcConditionManager* _condition_manager;
  /**
   * @definition
   *  _layouts : describe all shapes for all nets in all layers
   * @param
   *  LayoutType : cut | routing
   *  std::map<std::string, DrcEngineLayout*> - std::string : layer
   *  std::map<std::string, DrcEngineLayout*> - DrcEngineLayout : describe all shapes for all nets in all layers
   */
  std::map<LayoutType, std::map<std::string, DrcEngineLayout*>> _layouts;
  /**
   *  _scanline : scanline matrix
   */
  std::map<LayoutType, std::map<std::string, DrcEngineScanline*>> _scanline_matrix;
  // DrcEngineCheck* _engine_check = nullptr;
};

}  // namespace idrc