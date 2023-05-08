#pragma once

#include "DbInterface.h"
#include "DesignCalculator.h"
#include "EstimateParasitics.h"
#include "GDSwriter.h"

#include "ids.hpp"

namespace ito {

class ViolationOptimizer {
 public:
  ViolationOptimizer(DbInterface *dbinterface);
  ~ViolationOptimizer() { delete _parasitics_estimator; }
  ViolationOptimizer(const ViolationOptimizer &other) = delete;
  ViolationOptimizer(ViolationOptimizer &&other) = delete;

  void fixViolations();
  void fixViolations(const char *net_name);

 private:
  void initBuffer();
  void repairViolations(Net *net, StaVertex *drvr, bool check_slew, bool check_cap,
                        int &repair_count, int &slew_violations, int &cap_violations,
                        int &length_violations);

  void checkViolations();

  void fixViolations(RoutingTree *tree, int curr_pt, int prev_pt, Net *net, float max_cap,
                     int level,
                     // Return values.
                     int   &wire_length, // dbu
                     float &pin_cap, DesignObjSeq &load_pins);

  void fixLargeNet(Net *net, int max_fanout, LibertyCell *insert_buf_cell);

  RoutingTree *makeClusterTree(Net *net);

  template <class T1, class T2>
  void determineFixSide(T1 max_numb, T2 left, T2 middle, T2 right, bool &fix_left,
                        bool &fix_middle, bool &fix_right);

  void insertBuffer(int x, int y, Net *net, LibertyCell *insert_buf_cell, int level,
                    int &wire_length, float &cap, DesignObjSeq &load_pins);

  void checkFanoutViolation(DesignObject *drvr_pin,
                            // return values
                            double &max_fanout, int &fanout_violations,
                            bool &repair_fanout, vector<int> &fanouts);

  void checkCapacitanceViolation(DesignObject *drvr_pin,
                                 // return values
                                 double &max_drvr_cap, int &cap_violations,
                                 bool &repair_cap);

  void   checkSlewViolation(DesignObject *drvr_pin,
                            // return values
                            double &max_drvr_cap, int &slew_violations, bool &repair_slew);
  double calcLoadCap(LibertyPort *drvr_port, double slew);
  double calcSlewDiff(LibertyPort *drvr_port, double target_slew, double load_cap);
  void   calcGateRiseFallDelays(LibertyPort *drvr_port, float load_cap,
                                // return values.
                                Delay delays[], Slew slews[]);
  void   gateRiseFallDelay(TransType rf, LibertyArc *arc, float load_cap,
                           // return values.
                           Delay delays[], Slew slews[]);

  void setLocation(Instance *inst, int x, int y);

  void increDesignArea(float delta) { _db_interface->increDesignArea(delta); }

  bool hasMultipleOutputs(Instance *inst);

  bool repowerInstance(Pin *drvr_pin);
  bool repowerInstance(Instance *inst, LibertyCell *replace);

  double calcBufferDelay(LibertyCell *buffer_cell, float load);

  int portFanoutLoadNum(LibertyPort *port);

  bool netConnectToPort(Net *net);

  DbInterface     *_db_interface;
  TimingEngine    *_timing_engine;
  TimingDBAdapter *_db_adapter;

  bool _check_fanout = true;
  bool _check_cap = true;
  bool _check_slew = true;
  bool _check_length = true;

  vector<float>        _slew_record;
  vector<const char *> _fanout_vio_net;

  int _resize_instance_count;
  int _inserted_buffer_count;

  // to name the instance
  int _insert_instance_index = 1;
  // to name the net
  int _make_net_index = 1;

  int _dbu;

  static int _rise;
  static int _fall;

  // If there are still a violation nets, the secondary fix is performed.
  vector<Net *> _still_violation_net;

  // Instances with multiple output ports that have been resized.
  std::set<Instance *> _resized_multi_output_insts;

  EstimateParasitics *_parasitics_estimator;

  LibertyCell *_insert_buffer_cell;

  friend class SetupOptimizer;
  friend class HoldOptimizer;
};

} // namespace ito