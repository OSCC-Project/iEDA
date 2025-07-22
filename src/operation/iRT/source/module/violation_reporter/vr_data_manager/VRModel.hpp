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

#include "VRNet.hpp"

namespace irt {

class VRModel
{
 public:
  VRModel() = default;
  ~VRModel() = default;
  // getter
  std::vector<VRNet>& get_vr_net_list() { return _vr_net_list; }
  // setter
  void set_vr_net_list(const std::vector<VRNet>& vr_net_list) { _vr_net_list = vr_net_list; }

 private:
  std::vector<VRNet> _vr_net_list;
};

}  // namespace irt
