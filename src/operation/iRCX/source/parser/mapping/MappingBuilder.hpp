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
#include <unordered_map>
namespace ircx {
namespace parser {

class MappingBuilder
{
 public:
  MappingBuilder() = default;
  ~MappingBuilder() = default;

  const std::unordered_map<std::string, std::string>& design_to_process_layer_names() const {
    return design_to_process_layer_names_;
  }
  const std::unordered_map<std::string, std::string>& process_to_design_layer_names() const {
    return process_to_design_layer_names_;
  }

  void clear();
  bool read(const std::string& mappingPath);

 private:
  std::unordered_map<std::string, std::string> design_to_process_layer_names_;
  std::unordered_map<std::string, std::string> process_to_design_layer_names_;
};

}  // namespace parser
}  // namespace ircx
