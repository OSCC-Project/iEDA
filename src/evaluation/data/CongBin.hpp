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
#ifndef SRC_EVALUATOR_SOURCE_CONGESTION_DATABASE_CONGBIN_HPP_
#define SRC_EVALUATOR_SOURCE_CONGESTION_DATABASE_CONGBIN_HPP_

#include <memory>

#include "CongInst.hpp"
#include "CongNet.hpp"
#include "builder.h"

namespace eval {
class CongBin
{
 public:
  CongBin() : _x(0), _y(0), _lx(0), _ly(0), _ux(0), _uy(0) {}
  CongBin(int x, int y, int lx, int ly, int ux, int uy) : _x(x), _y(y), _lx(lx), _ly(ly), _ux(ux), _uy(uy) {}
  ~CongBin() { reset(); };

  // getter
  int get_x() const { return _x; }
  int get_y() const { return _y; }
  int get_lx() const { return _lx; }
  int get_ly() const { return _ly; }
  int get_ux() const { return _ux; }
  int get_uy() const { return _uy; }
  int64_t get_area() const { return static_cast<int64_t>(_ux - _lx) * static_cast<int64_t>(_uy - _ly); }

  int get_pin_num() const { return _pin_num; }
  double get_inst_density() const { return _inst_density; }
  double get_net_cong() const { return _net_cong; }

  int get_average_wire_width() const { return _average_wire_width; }
  int get_horizontal_capacity() const { return _horizontal_capacity; }
  int get_vertical_capacity() const { return _vertical_capacity; }

  std::vector<CongInst*> get_inst_list() const { return _inst_list; }
  std::vector<CongNet*> get_net_list() const { return _net_list; }

  // setter
  void set_pin_num(const int& pin_num) { _pin_num = pin_num; }
  void set_inst_density(const double& density) { _inst_density = density; }
  void set_net_cong(const double& bbox_cong) { _net_cong = bbox_cong; }

  void set_average_wire_width(const int& average_wire_width) { _average_wire_width = average_wire_width; }
  void set_horizontal_capacity(const int& horizontal_capacity) { _horizontal_capacity = horizontal_capacity; }
  void set_vertical_capacity(const int& vertical_capacity) { _vertical_capacity = vertical_capacity; }

  void add_inst(CongInst* inst) { _inst_list.push_back(inst); }
  void add_net(CongNet* net) { _net_list.push_back(net); }
  void increPinNum() { _pin_num++; }
  void increNetCong(const double& net_cong) { _net_cong += net_cong; }
  void reset();

 private:
  int _x;
  int _y;
  int _lx;
  int _ly;
  int _ux;
  int _uy;

  int _pin_num;
  double _inst_density;
  double _net_cong;

  int _average_wire_width;
  int _horizontal_capacity;
  int _vertical_capacity;

  std::vector<CongInst*> _inst_list;
  std::vector<CongNet*> _net_list;
};

inline void CongBin::reset()
{
  _pin_num = 0;
  _inst_density = 0.0;
  _net_cong = 0.0;
}

class CongGrid
{
 public:
  CongGrid() : _lx(0), _ly(0), _bin_cnt_x(0), _bin_cnt_y(0), _bin_size_x(0), _bin_size_y(0) {}
  CongGrid(int lx, int ly, int binCntX, int binCntY, int binSizeX, int binSizeY)
      : _lx(lx), _ly(ly), _bin_cnt_x(binCntX), _bin_cnt_y(binCntY), _bin_size_x(binSizeX), _bin_size_y(binSizeY)
  {
  }
  ~CongGrid()
  {
    _lx = _ly = 0;
    _bin_cnt_x = _bin_cnt_y = 0;
    _bin_size_x = _bin_size_y = 0;
    _bin_list.clear();
    _bin_list.shrink_to_fit();
  }

  // getter
  int get_lx() const { return _lx; }
  int get_ly() const { return _ly; }
  int get_ux() const { return _lx + _bin_cnt_x * _bin_size_x; }
  int get_uy() const { return _ly + _bin_cnt_y * _bin_size_y; }
  int get_bin_cnt_x() const { return _bin_cnt_x; }
  int get_bin_cnt_y() const { return _bin_cnt_y; }
  int get_bin_size_x() const { return _bin_size_x; }
  int get_bin_size_y() const { return _bin_size_y; }
  int get_routing_layers_number() const { return _routing_layers_number; }
  const std::vector<CongBin*>& get_bin_list() const { return _bin_list; }
  int get_track_num_h() const { return _track_num_h; }
  int get_track_num_v() const { return _track_num_v; }

  // setter
  void set_lx(const int& lx) { _lx = lx; }
  void set_ly(const int& ly) { _ly = ly; }
  void set_bin_cnt_x(const int& binCntX) { _bin_cnt_x = binCntX; }
  void set_bin_cnt_y(const int& binCntY) { _bin_cnt_y = binCntY; }
  void set_binCnt(const int& binCntX, const int& binCntY)
  {
    _bin_cnt_x = binCntX;
    _bin_cnt_y = binCntY;
  }
  void set_bin_size_x(const int& binSizeX) { _bin_size_x = binSizeX; }
  void set_bin_size_y(const int& binSizeY) { _bin_size_y = binSizeY; }
  void set_binSize(const int& binSizeX, const int& binSizeY) { _bin_size_x = binSizeX, _bin_size_y = binSizeY; }
  void set_routing_layers_number(const int& routing_layers_number) { _routing_layers_number = routing_layers_number; }

  void initBins();
  void initBins(idb::IdbLayers* idb_layer);
  void initTracksNum(idb::IdbLayers* idb_layer);

  int getRouteCapacity(const int& bin_size, idb::IdbLayerRouting* idb_layer_routing);
  int getWirePitch(idb::IdbLayerRouting* idb_layer_routing);
  int getWireWidth(idb::IdbLayerRouting* idb_layer_routing);

  std::pair<int, int> getMinMaxX(CongInst* inst);
  std::pair<int, int> getMinMaxY(CongInst* inst);
  std::pair<int, int> getMinMaxX(CongNet* net);
  std::pair<int, int> getMinMaxY(CongNet* net);

 private:
  int _lx;
  int _ly;
  int _bin_cnt_x;
  int _bin_cnt_y;
  int _bin_size_x;
  int _bin_size_y;
  int _routing_layers_number;
  std::vector<CongBin*> _bin_list;
  int32_t _track_num_h = 0;
  int32_t _track_num_v = 0;
};

}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_CONGESTION_DATABASE_CONGBIN_HPP_
