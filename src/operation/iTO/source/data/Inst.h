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

#include "Master.h"
#include "ids.hpp"
#include <string>
#include <vector>

using idb::IdbCellMaster;
using idb::IdbInstance;
using ito::Master;

namespace ito {
class Inst {
 public:
  Inst() = default;
  Inst(IdbInstance *inst);
  ~Inst() = default;
  Inst(const Inst &inst) = delete;
  Inst(Inst &&inst) = delete;

  Master *get_master() { return _master; }

 protected:
  Master *_master = nullptr;
};

} // namespace ito