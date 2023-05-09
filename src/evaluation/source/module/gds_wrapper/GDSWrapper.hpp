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
#ifndef SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRAPPER_GDSWRAPPER_HPP_
#define SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRAPPER_GDSWRAPPER_HPP_

#include <vector>

#include "GDSNet.hpp"

namespace eval {

class GDSWrapper
{
 public:
  GDSWrapper() = default;
  ~GDSWrapper() = default;

  void set_net_list(const std::vector<GDSNet*>& net_list) { _gds_net_list = net_list; }
  std::vector<GDSNet*>& get_net_list() { return _gds_net_list; }

 private:
  std::vector<GDSNet*> _gds_net_list;
};
}  // namespace eval

#endif  // SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRAPPER_GDSWRAPPER_HPP_
