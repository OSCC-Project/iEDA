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

#include "../module/placer/Placer.h"
#include "JsonParser.h"
#include "Layout.h"
#include "Reporter.h"
#include <set>

namespace ito {

using ito::ToConfig;

using namespace ista;

using Delay = double;
using Slew = Delay;
using Slack = Delay;
using Slacks = Slack[2][2];
using Required = Delay;
using ArcDelay = Delay;

using Level = int;
using VertexSeq = vector<StaVertex *>;
using LibertyCellSeq = vector<LibertyCell *>;
using DesignObjSeq = vector<ista::DesignObject *>;
using VertexSet = std::set<StaVertex *>;

using TgtSlews = std::array<Slew, 2>; // TransType: kFall / kRise;

using CellTargetLoadMap = std::map<LibertyCell *, float>;

const float kInf = 1E+30F;
#define REPORT_TO_TXT ;

class DbInterface {
 public:
  static DbInterface *get_db_interface(ToConfig *config, IdbBuilder *idb,
                                       TimingEngine *timing);
  static void         destroyDbInterface();

  TimingEngine *get_timing_engine() { return _timing_engine; }

  IdbBuilder *get_idb() { return _idb; }
  int         get_dbu() { return _dbu; }
  Rectangle   get_core() { return _core; }

  LibertyCellSeq get_buffer_cells() { return _buffer_cells; }
  VertexSeq      get_drvr_vertices() { return _drvr_vertices; }

  float  get_hold_slack_margin() { return _config->get_hold_slack_margin(); }
  float  get_setup_slack_margin() { return _config->get_setup_slack_margin(); }
  float  get_max_buffer_percent() { return _config->get_max_buffer_percent(); }
  float  get_max_utilization() { return _config->get_max_utilization(); }
  string get_output_def_file() { return _config->get_output_def_file(); }
  string get_gds_file() { return _config->get_gds_file(); }

  std::vector<std::string> get_drv_insert_buffers() {
    return _config->get_drv_insert_buffers();
  }
  std::vector<std::string> get_setup_insert_buffers() {
    return _config->get_setup_insert_buffers();
  }
  std::vector<std::string> get_hold_insert_buffers() {
    return _config->get_hold_insert_buffers();
  }

  int get_number_passes_allowed_decreasing_slack() {
    return _config->get_number_passes_allowed_decreasing_slack();
  }
  int get_rebuffer_max_fanout() { return _config->get_rebuffer_max_fanout(); }
  int get_split_load_min_fanout() { return _config->get_split_load_min_fanout(); }

  TgtSlews get_target_slews() { return _target_slews; }

  Reporter *report() { return _reporter; }
  Placer   *placer() { return _placer; }

  void increDesignArea(float delta) { _design_area += delta; }

  bool overMaxArea();

  bool isLinkCell(LibertyCell *cell);

  CellTargetLoadMap *get_cell_target_load_map() { return _cell_target_load_map; }

  LibertyCell *get_lowest_drive_buffer() { return _lowest_drive_buffer; }

  int &make_net_index() { return _make_net_index; }
  int &make_instance_index() { return _make_instance_index; }

 private:
  DbInterface(ToConfig *config) : _config(config){};
  ~DbInterface(){};

  void initData();

  void initDbData();

  void makeEquivCells();

  void findDrvrVertices();

  void findBufferCells();

  void findCellTargetLoads();
  void findBufferTargetSlews();
  void findBufferTargetSlews(LibertyCell *buffer,
                             // Return values.
                             Slew slews[], int counts[]);
  void getGateSlew(LibertyPort *port, TransType trans_type, LibertyArc *arc,
                   // Return values.
                   Slew slews[], int counts[]);

  void  findTargetLoad(LibertyCell *cell);
  void  findTargetLoad(LibertyCell *cell, LibertyArc *arc, TransType rf,
                       // return values
                       float &target_load_sum, int &arc_count);
  float findTargetLoad(LibertyCell *cell, LibertyArc *arc, TransType in_type,
                       TransType out_type);
  Slew  gateSlewDiff(TransType in_type, LibertyCell *cell, float load_cap, Slew in_slew,
                     Slew out_slew, LibertyArc *arc);

  static DbInterface *_db_interface;

  ToConfig     *_config = nullptr;
  IdbBuilder   *_idb = nullptr;
  TimingEngine *_timing_engine = nullptr;

  Reporter *_reporter = nullptr;
  Placer   *_placer = nullptr;

  int            _dbu;
  ito::Rectangle _core;
  double         _design_area = 0;
  Layout        *_layout = nullptr;

  VertexSeq      _drvr_vertices;
  LibertyCellSeq _buffer_cells;
  LibertyCell   *_lowest_drive_buffer;

  TgtSlews           _target_slews;
  CellTargetLoadMap *_cell_target_load_map{nullptr};

  static int _rise;
  static int _fall;

  static constexpr float _tgt_slew_load_cap_factor = 10.0;

  int _make_net_index = 0;
  int _make_instance_index = 0;
};
} // namespace ito
