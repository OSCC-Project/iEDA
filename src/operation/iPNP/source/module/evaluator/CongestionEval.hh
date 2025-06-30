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
 * @file CongestionEval.hh
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#pragma once

#include <iostream>
#include <list>
#include <string>
#include <vector>

#include "iPNPCommon.hh"

namespace ipnp {

class PNPConfig;

class CongestionEval
{
 public:
  CongestionEval() : _config(nullptr), _total_overflow_union(0) {}
  ~CongestionEval() = default;

  void evalEGR(idb::IdbBuilder* idb_builder);
  int32_t get_total_overflow_union() { return _total_overflow_union; }
  
  void set_config(PNPConfig* config) { _config = config; }
  
private:
  PNPConfig* _config;
  int32_t _total_overflow_union;
};

}  // namespace ipnp
