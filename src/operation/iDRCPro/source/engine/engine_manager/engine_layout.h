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
#include <stdint.h>

#include <map>

#include "engine_geometry.h"
#include "engine_sublayout.h"

namespace idb {
class IdbLayer;
}  // namespace idb

namespace idrc {
/**
 *  DrcEngineLayout definition : describe all shapes for all nets in one layer
 */

class DrcDataManager;

class DrcEngineLayout
{
 public:
  DrcEngineLayout(std::string layer) : _layer(layer) { _layout = new DrcEngineSubLayout(0); }
  ~DrcEngineLayout();

  std::map<int, DrcEngineSubLayout*>& get_sub_layouts() { return _sub_layouts; }
  DrcEngineSubLayout* get_sub_layout(int net_id);
  ieda_solver::EngineGeometry* get_net_engine(int net_id);
  DrcEngineSubLayout* get_layout() { return _layout; }

  uint64_t pointCount();

  bool addRect(int llx, int lly, int urx, int ury, int net_id);

  void combineLayout(DrcDataManager* data_manager);

 private:
  /**
   * _layer : layer name
   */
  std::string _layer;
  /**
   * int : net id, if equal to -1, sub layout is a kind of blockage
   * DrcEngineSubLayout* : sub layout ptr describe the net shapes
   */
  std::map<int, DrcEngineSubLayout*> _sub_layouts;
  DrcEngineSubLayout* _layout;
};

}  // namespace idrc