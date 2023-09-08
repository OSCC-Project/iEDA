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

#include <iostream>
#include <vector>

#include "CtsConfig.h"
#include "CtsDesign.hh"
#include "EvalNet.h"
#include "GDSPloter.h"

namespace icts {
using std::vector;

class Evaluator
{
 public:
  Evaluator() = default;
  Evaluator(const Evaluator&) = default;
  ~Evaluator() = default;

  void init();
  void evaluate();
  void update();

  double latency() const;
  double skew() const;
  double fanout() const;
  double slew() const;
  void statistics(const std::string& save_dir) const;
  int64_t wireLength() const;
  double dataCtsNetSlack() const;
  void plotPath(const string& inst, const string& file = "debug.gds") const;
  void plotNet(const string& net_name, const string& file = "debug.gds") const;

 private:
  void printLog();
  void transferData();
  void initLevel() const;
  void recursiveSetLevel(CtsNet* net) const;

  vector<EvalNet> _eval_nets;
  const int _default_size = 100;
};

}  // namespace icts