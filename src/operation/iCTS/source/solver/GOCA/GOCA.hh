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
 * @file GOCA.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <string>
#include <vector>

#include "ClockTopo.h"
#include "CtsInstance.h"
#include "Inst.hh"
#include "Node.hh"
namespace icts {
struct Assign
{
  int max_dist;    // max distance between centorid and inst
  int max_fanout;  // max fanout of a cluster
  double max_cap;  // max cap of a cluster
  double skew_bound;
  double ratio;  // clustering margin
};
/**
 * @brief Global Optimal Constraint Assignment
 *
 */
class GOCA
{
 public:
  GOCA() = delete;
  GOCA(const std::string& net_name, const std::vector<CtsInstance*>& instances) : _net_name(net_name), _instances(instances) {}

  ~GOCA() = default;
  // run
  void run();
  // get
  std::vector<ClockTopo> get_clk_topos() const { return _clock_topos; }

 private:
  // flow
  std::vector<Assign> globalAssign();
  std::vector<Inst*> assignApply(const std::vector<Inst*>& insts, const Assign& assign);
  std::vector<Inst*> topGuide(const std::vector<Inst*>& insts, const Assign& assign);
  Inst* netAssign(const std::vector<Inst*>& insts, const Assign& assign, const Point& level_center, const bool& shift = true);
  Net* saltOpt(const std::vector<Inst*>& insts, const Assign& assign);
  // interface
  void genClockTopo();
  // report
  void writeNetPy(Pin* root, const std::string& save_name = "net") const;
  void levelReport() const;
  // member
  std::string _net_name;
  std::vector<CtsInstance*> _instances;
  std::vector<ClockTopo> _clock_topos;
  std::vector<std::vector<Inst*>> _level_insts;
  std::vector<Net*> _nets;
  int _level = 1;
};
}  // namespace icts