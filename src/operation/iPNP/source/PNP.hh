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
 * @file PNP.hh
 * @author Xinhao li
 * @brief Top level file of PNP module.
 * @version 0.1
 * @date 2024-07-15
 */

#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "IREval.hh"
#include "PNPGridManager.hh"
#include "PNPIdbWrapper.hh"

namespace ipnp {

class PNP
{
 public:
  PNP();
  PNP(const std::string& config_file);
  ~PNP() = default;

  PNPGridManager get_initialized_network() { return _initialized_network; }
  PNPGridManager get_current_opt_network() { return _current_opt_network; }

  void init();
  void initIRAnalysis();
  void runSynthesis();
  void runOptimize();  // including calling Evaluator and modify PDN
  void runFastPlacer();
  void saveToIdb() { _idb_wrapper.saveToIdb(_current_opt_network); }
  void writeIdbToDef(std::string def_path);
  void runAnalysis();

  void connect_M2_M1();

  void run();  // According to the config. e.g. which Evaluator, which opt algorithm.

  // Set the output DEF file path
  void set_output_def_path(const std::string& path) { _output_def_path = path; }

 private:
  PNPGridManager _input_network;
  PNPGridManager _initialized_network;
  PNPGridManager _current_opt_network;

  PNPIdbWrapper _idb_wrapper;
  IREval _ir_eval;

  std::string _output_def_path;
};

}  // namespace ipnp
