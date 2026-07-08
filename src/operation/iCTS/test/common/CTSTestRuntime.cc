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
 * @file CTSTestRuntime.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-24
 * @brief Explicit runtime holder implementation for iCTS tests.
 */

#include "common/CTSTestRuntime.hh"

#include "Log.hh"
#include "flow/Flow.hh"

namespace icts_test::runtime {
namespace {

auto CurrentRuntimeStorage() -> std::unique_ptr<CTSTestRuntime>&
{
  static thread_local std::unique_ptr<CTSTestRuntime> runtime;
  return runtime;
}

}  // namespace

CTSTestRuntime::CTSTestRuntime() : _runtime(std::make_unique<icts::CTSRuntime>()), flow(std::make_unique<icts::Flow>(*(_runtime)))
{
}

CTSTestRuntime::~CTSTestRuntime() = default;

auto CTSTestRuntime::reset() const -> void
{
  (*_runtime).reset();
  (*flow).reset();
}

auto BeginCurrentRuntime() -> void
{
  auto& runtime = CurrentRuntimeStorage();
  LOG_FATAL_IF(runtime != nullptr) << "iCTS test runtime is already active for this thread.";
  runtime = std::make_unique<CTSTestRuntime>();
}

auto EndCurrentRuntime() -> void
{
  CurrentRuntimeStorage() = nullptr;
}

auto CurrentRuntime() -> icts::CTSRuntime&
{
  auto& runtime = CurrentRuntimeStorage();
  LOG_FATAL_IF(runtime == nullptr) << "iCTS test runtime is not active for this test case.";
  return *(runtime->_runtime);
}

}  // namespace icts_test::runtime
