// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file timing_io.h
 * @author qiming chu (me@emin.chat)
 * @brief Timing evaluation IO for tcl interface.
 * @version 0.1
 * @date 2025-06-02
 */

#ifndef EVALTIMING_H
#define EVALTIMING_H

#include <filesystem>
#include <fstream>
#include <string>

#include "idm.h"

namespace ieval {
class EvalTiming
{
 public:
  static bool runTimingEval(const std::string& routing_type = "FLUTE");

  static void printTimingResult();

  static void getConfig(idm::DataConfig& config) { config = dmInst->get_config(); }

  static void setOutputPath(const std::string& path);

 private:
  EvalTiming();
  ~EvalTiming();
  std::string _routing_type;
  static std::string _output_path;
};
}  // namespace ieval

#endif  // EVALTIMING_H
