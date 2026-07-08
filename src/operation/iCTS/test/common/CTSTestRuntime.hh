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
 * @file CTSTestRuntime.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-24
 * @brief Explicit iCTS test runtime fixtures.
 */

#pragma once

#ifndef ICTS_TEST_COMMON_CTS_TEST_RUNTIME_HH
#define ICTS_TEST_COMMON_CTS_TEST_RUNTIME_HH

#include <memory>

namespace icts {
struct CTSRuntime;
class Flow;
}  // namespace icts

namespace icts_test::runtime {

struct CTSTestRuntime
{
  std::unique_ptr<icts::CTSRuntime> _runtime;
  std::unique_ptr<icts::Flow> flow;

  CTSTestRuntime();
  ~CTSTestRuntime();

  CTSTestRuntime(const CTSTestRuntime& other) = delete;
  CTSTestRuntime(CTSTestRuntime&& other) = delete;
  auto operator=(const CTSTestRuntime& other) -> CTSTestRuntime& = delete;
  auto operator=(CTSTestRuntime&& other) -> CTSTestRuntime& = delete;

  auto reset() const -> void;
};

auto BeginCurrentRuntime() -> void;
auto EndCurrentRuntime() -> void;
auto CurrentRuntime() -> icts::CTSRuntime&;

}  // namespace icts_test::runtime

#endif  // ICTS_TEST_COMMON_CTS_TEST_RUNTIME_HH
