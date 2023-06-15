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
#include "Node.hh"

namespace icts {
/**
 * @brief Propagate type
 *       kNET: propagate until the node's type is not steiner
 *       kALL: propagate until tree's leaf
 *
 */
enum class PropagateType
{
  kNET,
  kALL
};
/**
 * @brief Timing propagator
 *       Propagate timing information from root to leaf
 *       Support: - fanout, cap, slew, delay calculation, propagation, update
 *                - based on net type (kNET) and all type (kALL)
 *
 */
class TimingPropagator
{
 public:
  TimingPropagator() = delete;
  ~TimingPropagator() = default;

  static void init();
  static void update(Node* node, const PropagateType& propagete_type = PropagateType::kNET);
  static void fanoutPropagate(Node* node, const PropagateType& propagete_type = PropagateType::kNET);
  static void capPropagate(Node* node, const PropagateType& propagete_type = PropagateType::kNET);
  static void slewPropagate(Node* node, const PropagateType& propagete_type = PropagateType::kNET);
  static void delayPropagate(Node* node, const PropagateType& propagete_type = PropagateType::kNET);
  static void updateFanout(Node* node);
  static void updateCap(Node* node);
  static void updateSlew(Node* node);
  static void updateDelay(Node* node);
  static uint16_t calcFanout(Node* node);
  static double calcCap(Node* node);
  static double calcIdealSlew(Node* node, Node* child);
  static double calcElmoreDelay(Node* node, Node* child);
  static double calcLen(Node* n1, Node* n2);
  static int64_t calcDist(Node* n1, Node* n2);

 private:
  static double _unit_cap;  // ohm
  static double _unit_res;  // pf
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