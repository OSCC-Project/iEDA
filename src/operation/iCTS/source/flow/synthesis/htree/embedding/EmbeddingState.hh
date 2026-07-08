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
 * @file EmbeddingState.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief H-tree embedding object naming and buffer-port lookup state.
 */

#pragma once

#include <cstddef>
#include <string>

#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/embedding/BufferPortTable.hh"

namespace icts::htree {

struct EmbeddingState
{
  HTree::Build* result = nullptr;
  BufferPortTable* port_table = nullptr;
  std::size_t edge_buffer_counter = 0U;
  std::size_t net_counter = 0U;
  std::string object_name_prefix;

  auto nextBufferName() -> std::string
  {
    const auto suffix = "htree_edge_buf_" + std::to_string(edge_buffer_counter++);
    return object_name_prefix.empty() ? "cts_" + suffix : object_name_prefix + "_" + suffix;
  }

  auto nextNetName() -> std::string
  {
    const auto suffix = "htree_net_" + std::to_string(net_counter++);
    return object_name_prefix.empty() ? "cts_" + suffix : object_name_prefix + "_" + suffix;
  }
};

}  // namespace icts::htree
