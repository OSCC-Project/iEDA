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
 * @file IdbConversion.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-30
 * @brief CTS iDB conversion entry.
 */

#pragma once

#include <cstddef>
#include <string>

namespace icts {

class Design;
class SchemaWriter;
class Wrapper;

struct IdbConversionInput
{
  Design* design = nullptr;
  Wrapper* wrapper = nullptr;
  SchemaWriter* reporter = nullptr;
};

struct IdbConversionSummary
{
  bool attempted = false;
  bool design_ready = false;
  bool success = false;
  std::size_t clock_count = 0U;
  bool idb_clock_tree_restored = false;
  std::string failed_clock;
  std::string failed_net;
  std::string failure_reason = "n/a";
};

class IdbConversion
{
 public:
  IdbConversion() = delete;

  static auto run(const IdbConversionInput& input) -> IdbConversionSummary;
};

}  // namespace icts
