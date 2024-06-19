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

using TODelay = double;
using TOSlew = TODelay;
using TOSlack = TODelay;
using TOSlacks = TOSlack[2][2];
using TORequired = TODelay;
using TOArcDelay = TODelay;

using TOLevel = int;
using TOVertexSeq = vector<StaVertex *>;
using TOLibertyCellSeq = vector<LibertyCell *>;
using TODesignObjSeq = vector<ista::DesignObject *>;
using TOVertexSet = std::set<StaVertex *>;

using TOSlewTarget = std::array<TOSlew, 2>; // TransType: kFall / kRise;

using TOLibCellLoadMap = std::map<LibertyCell *, float>;

const float kInf = 1E+30F;
#define REPORT_TO_TXT ;

struct EvalData
{
  std::string name;
  double initial_wns;
  double initial_tns;
  double initial_freq;
};

class DbInterface {
 public:
  static DbInterface *get_db_interface(ToConfig *config, IdbBuilder *idb,
                                       TimingEngine *timing);
  static void         destroyDbInterface();

  TimingEngine *get_timing_engine() { return _timing_engine; }

  IdbBuilder *get_idb() { return _idb; }
  int         get_dbu() { return _dbu; }
  Rectangle   get_core() { return _core; }

  TOLibertyCellSeq get_buffer_cells() { return _available_buffer_cells; }
  TOVertexSeq      get_drvr_vertices() { return _drvr_vertices; }

  float  get_hold_target_slack() { return _config->get_hold_target_slack(); }
  float  get_setup_target_slack() { return _config->get_setup_target_slack(); }
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

  TOSlewTarget get_target_slews() { return _target_slews; }

  Reporter *report() { return _reporter; }
  Placer   *placer() { return _placer; }

  void increDesignArea(float delta) { _design_area += delta; }

  bool overMaxArea();

  bool canFindLibertyCell(LibertyCell *cell);

  TOLibCellLoadMap *get_cell_target_load_map() { return _cell_target_load_map; }

  LibertyCell *get_lowest_drive_buffer() { return _lowest_drive_buffer; }

  int &make_net_index() { return _make_net_index; }
  int &make_instance_index() { return _make_instance_index; }

  void set_eval_data();
  std::vector<EvalData> eval_data() { return _eval_data; }

 private:
  DbInterface(ToConfig *config) : _config(config){};
  ~DbInterface(){};

  void initData();

  void initDbData();

  void findEquivLibCells();

  void findDrvrVertices();

  void findBufferCells();

  void calcCellTargetLoads();
  void calcTargetSlewsForBuffer();
  void calcTargetSlewsForBuffer(LibertyCell *buffer,
                             TOSlew slews[], int counts[]);
  void getGateSlew(LibertyPort *port, TransType trans_type, LibertyArc *arc,
                   TOSlew slews[], int counts[]);

  void  calcTargetLoad(LibertyCell *cell);
  void   calcTargetLoad(LibertyArc *arc, TransType rf, float &target_load_sum,
                        int &arc_count);
  float  calcTargetLoad(LibertyArc *arc, TransType in_type, TransType out_type);
  TOSlew calcSlewDiffOfGate(TransType in_type, float load_cap, TOSlew in_slew,
                            TOSlew out_slew, LibertyArc *arc);

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

  TOVertexSeq      _drvr_vertices;
  TOLibertyCellSeq _available_buffer_cells;
  LibertyCell     *_lowest_drive_buffer;

  TOSlewTarget      _target_slews;
  TOLibCellLoadMap *_cell_target_load_map{nullptr};

  static int _rise;
  static int _fall;

  static constexpr float _slew_2_load_cap_factor = 10.0;

  int _make_net_index = 0;
  int _make_instance_index = 0;

  std::vector<EvalData> _eval_data;
};
} // namespace ito
