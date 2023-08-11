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
#include "CongBin.hpp"

#include <cmath>

namespace eval {

void CongGrid::initBins()
{
  int x = _lx, y = _ly;
  int idxX = 0, idxY = 0;
  int bin_num = _bin_cnt_x * _bin_cnt_y;
  _bin_list.reserve(bin_num);
  for (int i = 0; i < bin_num; ++i) {
    CongBin* bin = new CongBin(idxX, idxY, x, y, x + _bin_size_x, y + _bin_size_y);
    x += _bin_size_x;
    idxX += 1;
    if (x >= get_ux()) {
      y += _bin_size_y;
      x = _lx;
      idxY++;
      idxX = 0;
    }
    _bin_list.push_back(bin);
  }
}

void CongGrid::initBins(idb::IdbLayers* idb_layer)
{
  int x = _lx, y = _ly;
  int idxX = 0, idxY = 0;
  int bin_num = _bin_cnt_x * _bin_cnt_y;
  _bin_list.reserve(bin_num);
  for (int i = 0; i < bin_num; ++i) {
    CongBin* bin = new CongBin(idxX, idxY, x, y, x + _bin_size_x, y + _bin_size_y);
    x += _bin_size_x;
    idxX += 1;
    if (x >= get_ux()) {
      y += _bin_size_y;
      x = _lx;
      idxY++;
      idxX = 0;
    }
    // map track resource for each bin
    int sum_wire_width = 0;
    int horizontal_count = 0;
    int vertical_count = 0;
    int horizontal_capacity = 0;
    int vertical_capacity = 0;

    for (int i = 0; i < _routing_layers_number; i++) {
      idb::IdbLayerRouting* layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer->find_routing_layer(i));
      sum_wire_width += getWireWidth(layer);
      bool is_horizontal = layer->is_horizontal();
      if (is_horizontal) {
        horizontal_count++;
        horizontal_capacity += getRouteCapacity(_bin_size_y, layer);
      } else {
        vertical_count++;
        vertical_capacity += getRouteCapacity(_bin_size_x, layer);
      }
    }
    bin->set_average_wire_width(sum_wire_width / _routing_layers_number);
    bin->set_horizontal_capacity(horizontal_capacity / horizontal_count);
    bin->set_vertical_capacity(vertical_capacity / vertical_count);
    _bin_list.push_back(bin);
  }
}

void CongGrid::initTracksNum(idb::IdbLayers* idb_layer)
{
  for (int i = 0; i < _routing_layers_number; i++) {
    idb::IdbLayerRouting* layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer->find_routing_layer(i));
    bool is_horizontal = layer->is_horizontal();
    if (is_horizontal) {
      _track_num_h += layer->get_prefer_track_grid()->get_track_num();
    } else {
      _track_num_v += layer->get_prefer_track_grid()->get_track_num();
    }
  }
}

int CongGrid::getRouteCapacity(const int& bin_size, idb::IdbLayerRouting* idb_layer_routing)
{
  int capacity = 0;
  idb::IdbTrackGrid* track_grid = idb_layer_routing->get_prefer_track_grid();
  idb::IdbTrack* track = track_grid->get_track();
  capacity = bin_size / track->get_pitch();
  return capacity;
}

int CongGrid::getWirePitch(idb::IdbLayerRouting* idb_layer_routing)
{
  idb::IdbTrackGrid* track_grid = idb_layer_routing->get_prefer_track_grid();
  idb::IdbTrack* track = track_grid->get_track();
  return track->get_pitch();
}

int CongGrid::getWireWidth(idb::IdbLayerRouting* idb_layer_routing)
{
  return idb_layer_routing->get_width();
}

std::pair<int, int> CongGrid::getMinMaxX(CongInst* inst)
{
  int lower_idx = (inst->get_lx() - _lx) / _bin_size_x;
  int upper_idx = (inst->get_ux() - _lx) / _bin_size_x;
  return std::make_pair(lower_idx, upper_idx);
}

std::pair<int, int> CongGrid::getMinMaxY(CongInst* inst)
{
  int lower_idx = (inst->get_ly() - _ly) / _bin_size_y;
  int upper_idx = (inst->get_uy() - _ly) / _bin_size_y;
  return std::make_pair(lower_idx, upper_idx);
}

std::pair<int, int> CongGrid::getMinMaxX(CongNet* net)
{
  int lower_idx = (net->get_lx() - _lx) / _bin_size_x;
  int upper_idx = (net->get_ux() - _lx) / _bin_size_x;
  return std::make_pair(lower_idx, upper_idx);
}

std::pair<int, int> CongGrid::getMinMaxY(CongNet* net)
{
  int lower_idx = (net->get_ly() - _ly) / _bin_size_y;
  int upper_idx = (net->get_uy() - _ly) / _bin_size_y;
  return std::make_pair(lower_idx, upper_idx);
}

}  // namespace eval
