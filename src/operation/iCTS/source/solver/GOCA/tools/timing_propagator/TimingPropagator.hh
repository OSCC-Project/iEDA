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
/**
 * @file TimingPropagator.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include "CtsCellLib.h"
#include "Inst.hh"
#include "Net.hh"
#include "Node.hh"
#include "Pin.hh"

namespace icts {
/**
 * @brief Timing propagator
 *       Propagate timing information from root to leaf
 *       Support: - cap, slew, delay calculation, propagation, update
 *
 */
class TimingPropagator
{
 public:
  TimingPropagator() = delete;
  ~TimingPropagator() = default;

  static void init();
  // net based
  static Net* genNet(const std::string& net_name, Pin* driver_pin, const std::vector<Pin*>& load_pins = {});
  static void updateLoads(Net* net);
  static void update(Net* net);
  static void netLenPropagate(Net* net);
  static void capPropagate(Net* net);
  static void slewPropagate(Net* net);
  static void cellDelayPropagate(Net* net);
  static void wireDelayPropagate(Net* net);
  // inst based
  static void updateCellDelay(Inst* inst);
  // node based
  static void updateNetLen(Node* node);
  static void updateCapLoad(Node* node);
  static void updateSlewIn(Node* node);
  static void updateWireDelay(Node* node);
  static double calcNetLen(Node* node);
  static double calcCapLoad(Node* node);
  static double calcIdealSlew(Node* node, Node* child);
  static double calcElmoreDelay(Node* node, Node* child);
  static double calcLen(Node* parent, Node* child);
  static int64_t calcDist(Node* parent, Node* child);
  static double calcLen(const Point& p1, const Point& p2);
  static int64_t calcDist(const Point& p1, const Point& p2);

  static double calcSkew(Node* node);
  static bool skewFeasible(Node* node, const std::optional<double>& skew_bound = std::nullopt);
  // pin based
  static void initLoadPinDelay(Pin* pin, const bool& by_cell = false);

  // info getter
  static double getUnitCap() { return _unit_cap; }
  static double getUnitRes() { return _unit_res; }
  static double getSkewBound() { return _skew_bound; }
  static int getDbUnit() { return _db_unit; }
  static double getMaxBufTran() { return _max_buf_tran; }
  static double getMaxSinkTran() { return _max_sink_tran; }
  static double getMaxCap() { return _max_cap; }
  static int getMaxFanout() { return _max_fanout; }
  static double getMaxLength() { return _max_length; }
  static double getMinInsertDelay() { return _min_insert_delay; }
  static icts::CtsCellLib* getMinSizeLib() { return _delay_libs.front(); }
  static icts::CtsCellLib* getMaxSizeLib() { return _delay_libs.back(); }
  static std::vector<icts::CtsCellLib*> getDelayLibs() { return _delay_libs; }

 private:
  static double _unit_cap;  // pf
  static double _unit_res;  // ohm
  static double _skew_bound;
  static int _db_unit;
  static double _max_buf_tran;
  static double _max_sink_tran;
  static double _max_cap;
  static int _max_fanout;
  static double _max_length;
  static double _min_insert_delay;
  static std::vector<icts::CtsCellLib*> _delay_libs;
};

}  // namespace icts