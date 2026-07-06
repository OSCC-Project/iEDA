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

#include <string>

#include "config/CompareSpefConfig.hh"
#include "config/PlotSpefConfig.hh"

namespace ircx {

#define RCX_API_INST (ircx::RCXAPI::getInst())

class RCXAPI
{
 public:
  static auto getInst() -> RCXAPI&
  {
    static RCXAPI inst;
    return inst;
  }

  // Main RCX flow.
  static auto init(const std::string& config_file) -> bool;
  static auto run() -> bool;
  static auto report() -> bool;

  // Standalone RCX utilities.
  static auto compare_spef(compare_spef::Config config) -> bool;
  static auto dump_net_shape() -> bool;
  static auto plot_spef(plot_spef::Config config) -> bool;

  RCXAPI(const RCXAPI& other) = delete;
  RCXAPI(RCXAPI&& other) = delete;
  auto operator=(const RCXAPI& other) -> RCXAPI& = delete;
  auto operator=(RCXAPI&& other) -> RCXAPI& = delete;

 private:
  RCXAPI();
  ~RCXAPI() = default;
};

}  // namespace ircx
