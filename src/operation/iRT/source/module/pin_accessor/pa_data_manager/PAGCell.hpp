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

#include "PAGCellId.hpp"
#include "SpaceRegion.hpp"

namespace irt {

class PAGCell : public SpaceRegion
{
 public:
  PAGCell() = default;
  ~PAGCell() = default;
  // getter
  PAGCellId& get_pa_gcell_id() { return _pa_gcell_id; }
  // setter
  void set_pa_gcell_id(const PAGCellId& pa_gcell_id) { _pa_gcell_id = pa_gcell_id; }
  // function

 private:
  PAGCellId _pa_gcell_id;
};

}  // namespace irt
