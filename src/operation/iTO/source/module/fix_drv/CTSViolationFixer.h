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

#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "DbInterface.h"
#include "EstimateParasitics.h"
#include "Utility.h"
#include "ctime"

#include "ids.hpp"

using std::array;
using std::map;
using std::string;
using std::vector;
namespace ito {

using idb::IdbNet;

using ito::hashIncr;

using Delay = double;
using Slew = Delay;
using Slack = Delay;
using Slacks = Slack[2][2];
using Required = Delay;
using ArcDelay = Delay;
using TgtSlews = array<Slew, 2>; // TransType: kFall / kRise;
using LibertyCellSeq = vector<LibertyCell *>;

using DesignObjSeq = std::vector<ista::DesignObject *>;

struct Edge {
  Edge(int father, int child) : _father(father), _child(child) {}
  ~Edge() = default;
  int _father;
  int _child;
};

class EdgeHash {
 public:
  size_t operator()(const Edge &edg) const {
    size_t hash = 5381;
    hashIncr(hash, edg._father);
    hashIncr(hash, edg._child);
    return hash;
  }
};

class EdgeEqual {
 public:
  bool operator()(const Edge &edg1, const Edge &edg2) const {
    return edg1._father == edg2._father && edg1._child == edg2._child;
  }
};

class Tree {
 public:
  Tree(int size) {
    _left.resize(size, _null_pt);
    _middle.resize(size, _null_pt);
    _right.resize(size, _null_pt);
  }
  ~Tree() = default;

  int left(int idx) const { return _left[idx]; }
  int middle(int idx) const { return _middle[idx]; }
  int right(int idx) const { return _right[idx]; }

  void set_drvr_id(int id) { _drvr_id = id; }
  int  get_drvr_id() const { return _drvr_id; }

  void add_edge(Edge edg);

  DesignObject *get_pin(int id);

  Point get_location(int id);

  void idToDesignObject(int id, DesignObject *obj) { _id_to_design_obj[id] = obj; }

  void idToLocation(int id, Point loc) { _id_to_location[id] = loc; }

  void       printTree();
  static int _null_pt;

 private:
  vector<int> _left;
  vector<int> _middle;
  vector<int> _right;

  int _drvr_id = -1;

  map<int, Point> _id_to_location;

  map<int, DesignObject *> _id_to_design_obj;
};

class CTSViolationFixer {
 public:
  static CTSViolationFixer *get_cts_violation_fixer(DbInterface *dbInterface);
  static CTSViolationFixer *get_cts_violation_fixer(idb::IdbBuilder    *idb = nullptr,
                                                    ista::TimingEngine *timing = nullptr);

  static void destroyCTSViolationFixer();

  // return  idb::Net
  // void fixTiming(idbNet, ClockTopos clk_topo);
  std::vector<IdbNet *> fixTiming(IdbNet *idb_net, Tree *topo);

 private:
  CTSViolationFixer() = delete;
  CTSViolationFixer(idb::IdbBuilder *idb, ista::TimingEngine *timing);
  CTSViolationFixer(DbInterface *dbInterface);
  ~CTSViolationFixer() = default;
  CTSViolationFixer(const CTSViolationFixer &other) = delete;
  CTSViolationFixer(CTSViolationFixer &&other) = delete;

  void checkViolation(Net *net, int &slew_violations, int &cap_violations,
                      int &fanout_violations);

  void fixViolations(Tree *tree, int curr_pt, int prev_pt, Net *net, float max_cap,
                     float max_fanout, int level,
                     // Return values.
                     // Remaining parasiics after repeater insertion.
                     int   &wire_length, // dbu
                     float &pin_cap, float &fanout, DesignObjSeq &load_pins);

  // Tree *clkTreeToTree(Net *sta_net, CtsNet *cnet);
  pair<int, int> selectBufferLocation(std::deque<Point> &segments, Point current_loc,
                                      Point prev_loc, double buf_dist, int length);

  void checkFanoutViolation(DesignObject *drvr_pin,
                            // return values
                            double &max_fanout, bool &repair_fanout);

  void checkCapacitanceViolation(DesignObject *drvr_pin,
                                 // return values
                                 double &max_drvr_cap, bool &repair_cap);

  void checkSlewViolation(DesignObject *drvr_pin,
                          // return values
                          double &max_drvr_cap, bool &repair_slew);

  double calcLoadCap(LibertyPort *drvr_port, double slew);
  double calcSlewDiff(LibertyPort *drvr_port, double target_slew, double load_cap);
  void   calcGateRiseFallDelays(LibertyPort *drvr_port, float load_cap,
                                // return values.
                                Delay delays[], Slew slews[]);
  void   gateRiseFallDelay(TransType rf, LibertyArc *arc, float load_cap,
                           // return values.
                           Delay delays[], Slew slews[]);

  int portFanoutLoadNum(LibertyPort *port);

  template <class T1, class T2>
  void determineFixSide(T1 max_numb, T2 left, T2 middle, T2 right, bool &fix_left,
                        bool &fix_middle, bool &fix_right);

  void insertCLKBuffer(int x, int y, Net *net, LibertyCell *insert_buf_cell, int level,
                       int &wire_length, float &cap, float &fanout,
                       DesignObjSeq &load_pins);

  void setLocation(Instance *inst, int x, int y);

  //   IdbAdapter *_cts_db = nullptr;
  // ClockTopos _clk_topos;
  static CTSViolationFixer *_cts_drv_fix;

  DbInterface *_db_interface;
  // CtsNet *_cts_net;
  TimingEngine          *_timing_engine;
  ista::TimingDBAdapter *_db_adapter = nullptr;

  ito::EstimateParasitics *_parasitics_estimator = nullptr;

  Placer *_placer = nullptr;

  int _dbu;

  TgtSlews       _targ_slews;
  LibertyCellSeq _buffer_cells;

  vector<int> _fanouts;

  // to name the instance
  int _insert_instance_index = 1;
  // to name the net
  int _make_net_index = 1;

  int _inserted_buffer_count = 0;

  std::vector<idb::IdbNet *> _net_for_estimate;

  vector<pair<string, double>> _store;

  static int _rise;
  static int _fall;

  bool _check_fanout;
  bool _check_cap;
  bool _check_slew;

  string _insert_clk_buf = "LVT_CLKBUFHDV8";
};
} // namespace ito
