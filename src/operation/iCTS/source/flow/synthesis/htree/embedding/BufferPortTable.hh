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
 * @file BufferPortTable.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief H-tree embedding buffer port lookup table.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <utility>

#include "io/Wrapper.hh"

namespace icts::htree {

class BufferPortTable
{
 public:
  explicit BufferPortTable(Wrapper& wrapper) : _wrapper(&wrapper) {}

  auto get(const std::string& cell_master) -> const std::pair<std::string, std::string>*
  {
    auto it = _ports_by_master.find(cell_master);
    if (it != _ports_by_master.end()) {
      return &it->second;
    }

    auto [input_pin, output_pin] = _wrapper->queryBufferPorts(cell_master);
    if (input_pin.empty() || output_pin.empty()) {
      return nullptr;
    }

    auto [inserted_it, inserted] = _ports_by_master.emplace(cell_master, std::make_pair(std::move(input_pin), std::move(output_pin)));
    (void) inserted;
    return &inserted_it->second;
  }

 private:
  Wrapper* _wrapper = nullptr;
  std::unordered_map<std::string, std::pair<std::string, std::string>> _ports_by_master;
};

}  // namespace icts::htree
