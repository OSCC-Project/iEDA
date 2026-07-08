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
 * @file ScopedLogFile.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Scoped test log-file redirection helper.
 */

#pragma once

#include <filesystem>
#include <string>

namespace icts_test::common::logging {

class ScopedLogFile
{
 public:
  explicit ScopedLogFile(const std::filesystem::path& path, const std::string& run_title = "iCTS Test Report");
  ~ScopedLogFile();

  ScopedLogFile(const ScopedLogFile&) = delete;
  auto operator=(const ScopedLogFile&) -> ScopedLogFile& = delete;
  ScopedLogFile(ScopedLogFile&&) = delete;
  auto operator=(ScopedLogFile&&) -> ScopedLogFile& = delete;
};

}  // namespace icts_test::common::logging
