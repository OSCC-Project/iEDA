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

#include "GPStruct.hpp"

namespace idrc {

class GPGDS
{
 public:
  GPGDS() = default;
  GPGDS(std::string top_name) { _top_name = top_name; }
  ~GPGDS() = default;
  // getter
  std::string& get_top_name() { return _top_name; }
  std::vector<GPStruct>& get_struct_list() { return _struct_list; }
  // setter

  // function
  void addStruct(GPStruct& gp_struct) { _struct_list.push_back(gp_struct); }

 private:
  std::string _top_name = "top";
  std::vector<GPStruct> _struct_list;
};

}  // namespace idrc
