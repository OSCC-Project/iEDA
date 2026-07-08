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
 * @file LogText.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-27
 * @brief Shared text helpers for iCTS log contract tests.
 */

#pragma once

#include <string>

namespace icts_test::common::logging {

inline auto ExtractTextBlock(const std::string& content, const std::string& title) -> std::string
{
  const auto title_position = content.find(title);
  if (title_position == std::string::npos) {
    return {};
  }

  const auto line_start_position = content.rfind('\n', title_position);
  const auto block_start_position = line_start_position == std::string::npos ? title_position : line_start_position + 1U;
  const auto block_end_position = content.find("\n\n", title_position);
  if (block_end_position == std::string::npos) {
    return content.substr(block_start_position);
  }
  return content.substr(block_start_position, block_end_position - block_start_position);
}

}  // namespace icts_test::common::logging
