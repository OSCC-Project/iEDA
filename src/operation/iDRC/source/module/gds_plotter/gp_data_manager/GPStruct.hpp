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

#include "GPBoundary.hpp"
#include "GPPath.hpp"
#include "GPText.hpp"

namespace idrc {

class GPStruct
{
 public:
  GPStruct() = default;
  GPStruct(std::string name)
  {
    _name = name;
    _alias_name = name;
  }
  GPStruct(std::string name, std::string alias_name)
  {
    _name = name;
    _alias_name = alias_name;
  }
  ~GPStruct() = default;
  // getter
  std::string& get_name() { return _name; }
  std::string& get_alias_name() { return _alias_name; }
  std::vector<GPBoundary>& get_boundary_list() { return _boundary_list; }
  std::vector<GPPath>& get_path_list() { return _path_list; }
  std::vector<GPText>& get_text_list() { return _text_list; }
  std::vector<std::string>& get_sref_name_list() { return _sref_name_list; }
  // setter
  void set_alias_name(const std::string& alias_name) { _alias_name = alias_name; }
  // function
  void push(const GPBoundary& gp_boundary) { _boundary_list.push_back(gp_boundary); }
  void push(const GPPath& gp_path) { _path_list.push_back(gp_path); }
  void push(const GPText& gp_text) { _text_list.push_back(gp_text); }
  void push(const std::string& sref_name) { _sref_name_list.push_back(sref_name); }

 private:
  std::string _name;
  std::string _alias_name;
  std::vector<GPBoundary> _boundary_list;
  std::vector<GPPath> _path_list;
  std::vector<GPText> _text_list;
  std::vector<std::string> _sref_name_list;
};

}  // namespace idrc
