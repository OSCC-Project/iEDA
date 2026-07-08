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
 * @file Instantiation.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS instantiation entry facade.
 */

#pragma once

#include <cstddef>
#include <string>

namespace icts {

class Design;
class SchemaWriter;
class Wrapper;

struct InstantiationInput
{
  Design* design = nullptr;
  Wrapper* wrapper = nullptr;
  SchemaWriter* reporter = nullptr;
};

struct InstantiationSummary
{
  bool attempted = false;
  bool design_ready = false;
  bool success = false;
  bool design_conversion_done = false;
  bool idb_conversion_done = false;
  std::size_t clock_count = 0U;
  std::string failure_reason = "n/a";
};

class Instantiation
{
 public:
  Instantiation() = delete;

  static auto run(const InstantiationInput& input) -> InstantiationSummary;
};

}  // namespace icts
