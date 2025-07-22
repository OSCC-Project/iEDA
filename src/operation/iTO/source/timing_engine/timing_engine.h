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

#include <list>
#include <map>
#include <string>
#include <vector>

#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "define.h"

using namespace ista;

const int TYPE_RISE = (int) ista::TransType::kRise - 1;
const int TYPE_FALL = (int) ista::TransType::kFall - 1;

namespace ito {
using namespace std;

#define timingEngine ToTimingEngine::getInstance()

struct EvalData
{
  std::string name;
  double initial_wns;
  double initial_tns;
  double initial_freq;
};

class TOLibCellLoadMap;
class ToTimingEngine
{
 public:
  static ToTimingEngine* getInstance()
  {
    if (nullptr == _instance) {
      _instance = new ToTimingEngine();
    }
    return _instance;
  }

  void initEngine();

  /// getter
  TOLibertyCellSeq& get_buffer_cells() { return _available_buffer_cells; }
  TOVertexSeq& get_driver_vertices() { return _driver_vertices; }
  ista::TimingEngine* get_sta_engine() { return _timing_engine; }
  ista::TimingIDBAdapter* get_sta_adapter() { return dynamic_cast<ista::TimingIDBAdapter*>(_timing_engine->get_db_adapter()); }

  TOLibCellLoadMap* get_target_map();
  LibCell* get_buf_lowest_driver_res() { return _buf_lowest_driver_res; }
  TOSlewTarget& get_target_slews() { return _target_slews; }
  std::optional<TOSlack> getNodeWorstSlack(StaVertex* node);
  StaSeqPathData* getNodeWorstPath(StaVertex* node);
  double getWNS();

  ista::LibCell* get_drv_buffer();

  std::vector<EvalData> eval_data() { return _eval_data; }

  /// setter
  void set_sta_engine(ista::TimingEngine* timing_engine) { _timing_engine = timing_engine; }

  void new_target_map();
  void set_buf_lowest_driver_res(LibCell* buf_lowest_driver_res) { _buf_lowest_driver_res = buf_lowest_driver_res; }

  bool canFindLibertyCell(LibCell* cell);

  /// operator
  void set_eval_data();

  /// calculate
  void calcGateRiseFallDelays(TODelay rise_fall_delay[], float cap_load, LibPort* driver_port);
  void calcGateRiseFallSlews(TOSlew rise_fall_slew[], float cap_load, LibPort* driver_port);
  void calGateRiseFallDelay(TODelay rise_fall_delay[], float cap_load, TransType rf, LibArc* arc);
  void calGateRiseFallSlew(TOSlew rise_fall_slew[], float cap_load, TransType rf, LibArc* arc);
  double calcDelayOfBuffer(float cap_load, LibCell* buffer_cell);

  float calcSetupDelayOfBuffer(float cap_load, LibCell* buffer_cell);
  float calcSetupDelayOfBuffer(float cap_load, TransType rf, LibCell* buffer_cell);
  float calcSetupDelayOfGate(float cap_load, LibPort* driver_port);
  float calcSetupDelayOfGate(float cap_load, TransType rf, LibPort* driver_port);

  /// setup calculate

  /// instance process
  bool repowerInstance(Pin* driver_pin);
  bool repowerInstance(ista::LibCell* repower_size, ista::Instance* repowered_inst);
  void placeInstance(int x, int y, ista::Instance* place_inst);

  /// slack calculate
  TOSlack getWorstSlack(StaVertex* vertex, AnalysisMode mode);
  TOSlack getWorstSlack(AnalysisMode mode);

  void refineRes(RctNode* node1, RctNode* node2, Net* net, double res = 1.0e-3, bool b_incre = false, double incre_cap = 0.0);

 private:
  static ToTimingEngine* _instance;
  ista::TimingEngine* _timing_engine = nullptr;

  /// data structure
  TOVertexSeq _driver_vertices;
  TOLibertyCellSeq _available_buffer_cells;
  LibCell* _buf_lowest_driver_res;

  TOSlewTarget _target_slews;
  TOLibCellLoadMap* _target_map = nullptr;

  std::vector<EvalData> _eval_data;

  /// constuctor
  ToTimingEngine();
  ~ToTimingEngine();
};

}  // namespace ito