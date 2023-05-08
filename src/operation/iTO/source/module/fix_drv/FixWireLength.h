#pragma once

#include "DbInterface.h"
#include "RoutingTree.h"
// #include "DesignCalculator.h"
// #include "EstimateParasitics.h"

#include "ids.hpp"

namespace ito {
class FixWireLength {
 public:
  FixWireLength(ista::TimingEngine *timing);
  ~FixWireLength() = default;

  void fixMaxLength(int max_length);

  void set_insert_buffer(LibertyCell *insert_buf);

  void fixMaxLength(Net *net, int max_length, bool fix = false);
 private:

  void fixMaxLength(RoutingTree *tree, int curr_pt, int prev_pt, Net *net, int max_length,
                    int level,
                    // Return values.
                    int &wire_length, DesignObjSeq &load_pins);

  template <class T1, class T2>
  void determineFixSide(T1 max_numb, T2 left, T2 middle, T2 right, bool &fix_left,
                        bool &fix_middle, bool &fix_right);

  void insertBuffer(int x, int y, Net *net, LibertyCell *insert_buf_cell, int level,
                    int &wire_length, DesignObjSeq &load_pins);

  void setLocation(Instance *inst, int x, int y);

  TimingEngine    *_timing_engine;
  TimingDBAdapter *_db_adapter;

  LibertyCell *_insert_buffer_cell;
  // to name the instance
  int _insert_instance_index = 1;
  // to name the net
  int _make_net_index = 1;
};
} // namespace ito
