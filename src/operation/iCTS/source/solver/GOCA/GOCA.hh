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

#include "CtsInstance.hh"
#include "CtsNet.hh"
#include "CtsPin.hh"
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
  GOCA(const std::string& net_name, CtsPin* cts_driver, const std::vector<CtsPin*>& cts_pins)
      : _net_name(net_name), _cts_driver(cts_driver), _cts_pins(cts_pins)
  {
  }

  ~GOCA() = default;
  // run
  void run();
  // get
  std::vector<Net*> get_solver_nets() const { return _nets; }

 private:
  // flow
  void init();
  void resolveSinks();
  void breakLongWire();
  std::vector<Assign> globalAssign();
  std::vector<Inst*> assignApply(const std::vector<Inst*>& insts, const Assign& assign);
  std::vector<Inst*> topGuide(const std::vector<Inst*>& insts, const Assign& assign);
  Inst* netAssign(const std::vector<Inst*>& insts, const Assign& assign, const Point& level_center, const bool& shift = true);
  Net* saltOpt(const std::vector<Inst*>& insts, const Assign& assign);

  // report
  void writeNetPy(Pin* root, const std::string& save_name = "net") const;
  void levelReport() const;
  // member
  std::string _net_name;
  CtsPin* _cts_driver;
  std::vector<CtsPin*> _cts_pins;
  std::vector<std::vector<Inst*>> _level_insts;
  std::vector<Pin*> _sink_pins;
  std::vector<Pin*> _top_pins;
  Pin* _driver = nullptr;
  std::vector<Net*> _nets;
  int _level = 1;
};
}  // namespace icts