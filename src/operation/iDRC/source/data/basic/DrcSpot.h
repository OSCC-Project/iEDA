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
#ifndef IDRC_SRC_DB_DRCSPOT_H_
#define IDRC_SRC_DB_DRCSPOT_H_

#include <vector>

#include "DrcEnum.h"
#include "DrcRect.h"
#include "DrcRectangle.h"
namespace idrc {
class DrcSpot
{
 public:
  DrcSpot() : _violation_type(ViolationType::kNone) { _spot_drc_rect_list.clear(); }
  DrcSpot(const DrcSpot& other)
  {
    _violation_type = other._violation_type;
    _spot_drc_rect_list = other._spot_drc_rect_list;
    _spot_drc_edge_list = other._spot_drc_edge_list;
    _eol_vio = other._eol_vio;
  }
  DrcSpot(DrcSpot&& other)
  {
    _violation_type = std::move(other._violation_type);
    _spot_drc_rect_list = std::move(other._spot_drc_rect_list);
    _spot_drc_edge_list = std::move(other._spot_drc_edge_list);
    _eol_vio = std::move(other._eol_vio);
  }
  ~DrcSpot() {}

  DrcSpot& operator=(const DrcSpot& other)
  {
    _violation_type = other._violation_type;
    _spot_drc_rect_list = other._spot_drc_rect_list;
    return *this;
  }
  DrcSpot& operator=(DrcSpot&& other)
  {
    _violation_type = std::move(other._violation_type);
    _spot_drc_rect_list = std::move(other._spot_drc_rect_list);
    return *this;
  }

  // setter
  // void set_layer_id(int layerId) { _layer_id = layerId; }
  void set_violation_type(ViolationType type) { _violation_type = type; }
  // void set_violation_box(const DrcRectangle<int>& violation_box) { _violation_box = violation_box; }
  // void add_spot_rect(const DrcRectangle<int>& rect) { _spot_rect_list.push_back(rect); }
  void add_spot_rect(DrcRect* drc_rect) { _spot_drc_rect_list.push_back(drc_rect); }
  void add_spot_edge(DrcEdge* drc_edge) { _spot_drc_edge_list.push_back(drc_edge); }
  void add_eol_vio(DrcRect* drc_rect, DrcEdge* drc_edge)
  {
    _eol_vio.first = drc_edge;
    _eol_vio.second = drc_rect;
  }
  // getter
  std::vector<DrcRect*>& get_spot_drc_rect_list() { return _spot_drc_rect_list; }
  std::pair<DrcEdge*,DrcRect*>& get_spot_eol_vio(){return _eol_vio;} 
  std::vector<DrcEdge*>& get_spot_drc_edge_list(){return _spot_drc_edge_list;}
  // int get_layer_id() const { return _layer_id; }
  ViolationType get_violation_type() const { return _violation_type; }


  // DrcRectangle<int> get_violation_box() { return _violation_box; }
  // std::vector<DrcRectangle<int>>& get_spot_rect_list() { return _spot_rect_list; }
  // function
  void clearSpotRects()
  {
    for (auto& drc_rect : _spot_drc_rect_list) {
      if (drc_rect->get_owner_type() == RectOwnerType::kSpotMark && drc_rect != nullptr) {
        delete drc_rect;
        drc_rect = nullptr;
      }
    }
    _spot_drc_rect_list.clear();
  }

 private:
  // int _layer_id;
  ViolationType _violation_type;
  // DrcRectangle<int> _violation_box;
  // std::vector<DrcRectangle<int>> _spot_rect_list;
  std::vector<DrcEdge*> _spot_drc_edge_list;
  std::vector<DrcRect*> _spot_drc_rect_list;
  std::pair<DrcEdge*, DrcRect*> _eol_vio;
};
}  // namespace idrc

#endif