#pragma once

#include "ViolationOptimizer.h"
#include "ids.hpp"

namespace ito {

class HoldOptimizer {
 public:
  HoldOptimizer(DbInterface *dbinterface);

  ~HoldOptimizer() {
    delete _parasitics_estimator;
    delete _violation_fixer;
  }
  HoldOptimizer(const HoldOptimizer &other) = delete;
  HoldOptimizer(HoldOptimizer &&other) = delete;

  // open functions
  void optimizeHold();

  void insertHoldDelay(string insert_buf_name, string pin_name, int insert_number = 1);

 private:
  int checkAndOptimizeHold(VertexSet end_points, LibertyCell *insert_buf_cell);

  void initBufferCell();

  void calcBufferCap();

  LibertyCell *findBufferWithMaxDelay();

  void findEndpointsWithHoldViolation(VertexSet end_points,
                                      // return values
                                      Slack &worst_slack, VertexSet &hold_violations);

  int  fixHoldPass(VertexSeq fanins, LibertyCell *insert_buffer_cell);

  void insertBufferDelay(StaVertex *drvr_vertex, int insert_number,
                         DesignObjSeq &load_pins, LibertyCell *insert_buffer_cell);

  void  vertexSlacks(StaVertex *vertex,
                     // return value
                     Slacks slacks);
  Slack calcSlackGap(StaVertex *vertex);

  void setLocation(Instance *inst, int x, int y);

  float getBufferHoldDelay(LibertyCell *buffer);

  VertexSet getEndPoints();

  VertexSet getFanins(VertexSet end_points);

  VertexSeq sortFanins(VertexSet fanins);

  Slack getWorstSlack(AnalysisMode mode);
  Slack getWorstSlack(StaVertex *vertex, AnalysisMode mode);

  void insertLoadBuffer(LibertyCell *load_buffer, StaVertex *drvr_vtx, int insert_num);
  void insertLoadBuffer(VertexSeq fanins);

  void reportWNSAndTNS();

  // data
  DbInterface     *_db_interface;
  TimingEngine    *_timing_engine;
  TimingDBAdapter *_db_adapter;

  EstimateParasitics *_parasitics_estimator;
  ViolationOptimizer *_violation_fixer;

  LibertyCellSeq _buffer_cells;

  int _dbu;

  float _slack_margin = 0.0;
  bool  _allow_setup_violation = true;
  int   _max_numb_insert_buf = 0;

  int _inserted_buffer_count = 0;
  int _inserted_load_buffer_count = 0;

  // to name the instance
  int _insert_instance_index = 1;
  int _insert_load_instance_index = 1;
  // to name the net
  int _make_net_index = 1;
  int _make_load_net_index = 1;

  static int _mode_max;
  static int _mode_min;
  static int _rise;
  static int _fall;

  vector<pair<double, LibertyCell *>> _buffer_cap_pair;
};

} // namespace ito